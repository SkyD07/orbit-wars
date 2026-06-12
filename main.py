"""
Orbit Wars development bot scaffold.

This file keeps the strategy small and puts reusable derived-data helpers up
front. The intent is to make future rule changes easy: add or change a helper,
then adjust choose_moves().
"""

import builtins
import math
from collections import defaultdict, namedtuple


BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
MAX_STEPS = 500

QUADRANT_KEYS = ("top_left", "top_right", "bottom_left", "bottom_right")
VERTICAL_HALF_KEYS = ("left", "right")
HORIZONTAL_HALF_KEYS = ("top", "bottom")

Planet = namedtuple("Planet", "id owner x y radius ships production")
Fleet = namedtuple("Fleet", "id owner x y angle from_planet_id ships")

if not hasattr(builtins, "_ORBIT_WARS_OPENING_GROUP_CACHE"):
    builtins._ORBIT_WARS_OPENING_GROUP_CACHE = {}
OPENING_GROUP_CACHE = builtins._ORBIT_WARS_OPENING_GROUP_CACHE


def read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def as_planets(obs):
    return [Planet(*p) for p in (read(obs, "planets", []) or [])]


def as_fleets(obs):
    return [Fleet(*f) for f in (read(obs, "fleets", []) or [])]


def build_context(obs):
    player = int(read(obs, "player", 0) or 0)
    step = int(read(obs, "step", 0) or 0)
    planets = as_planets(obs)
    fleets = as_fleets(obs)
    initial_planets = [Planet(*p) for p in (read(obs, "initial_planets", []) or [])]
    comet_ids = set(read(obs, "comet_planet_ids", []) or [])
    comets = read(obs, "comets", []) or []
    angular_velocity = float(read(obs, "angular_velocity", 0.0) or 0.0)

    context = {
        "player": player,
        "step": step,
        "remaining_steps": MAX_STEPS - step,
        "planets": planets,
        "fleets": fleets,
        "initial_planets": initial_planets,
        "initial_by_id": {p.id: p for p in initial_planets},
        "planet_by_id": {p.id: p for p in planets},
        "comet_ids": comet_ids,
        "comets": comets,
        "angular_velocity": angular_velocity,
        "my_planets": [p for p in planets if p.owner == player],
        "enemy_planets": [p for p in planets if p.owner not in (-1, player)],
        "neutral_planets": [p for p in planets if p.owner == -1],
        "my_fleets": [f for f in fleets if f.owner == player],
        "enemy_fleets": [f for f in fleets if f.owner != player],
    }
    context.update(build_spatial_groups(planets, fleets))
    context["opening_groups"] = get_opening_groups(context)
    context["start_region_value_report"] = {
        "quadrants": {
            "my_region": {
                "priority_target": context["opening_groups"].get("priority_target")
            },
            "priority_groups": context["opening_groups"].get("priority_groups", []),
            "rotating_groups": context["opening_groups"].get("rotating_groups", []),
        }
    }
    return context


def quadrant_key(x, y):
    if x < CENTER and y < CENTER:
        return "top_left"
    if x >= CENTER and y < CENTER:
        return "top_right"
    if x < CENTER and y >= CENTER:
        return "bottom_left"
    return "bottom_right"


def vertical_half_key(x, y):
    return "left" if x < CENTER else "right"


def horizontal_half_key(x, y):
    return "top" if y < CENTER else "bottom"


def empty_spatial_group(keys):
    return {key: {"planets": [], "fleets": []} for key in keys}


def build_spatial_groups(planets, fleets):
    quadrant_regions = empty_spatial_group(QUADRANT_KEYS)
    vertical_regions = empty_spatial_group(VERTICAL_HALF_KEYS)
    horizontal_regions = empty_spatial_group(HORIZONTAL_HALF_KEYS)

    for planet in planets:
        quadrant_regions[quadrant_key(planet.x, planet.y)]["planets"].append(planet)
        vertical_regions[vertical_half_key(planet.x, planet.y)]["planets"].append(planet)
        horizontal_regions[horizontal_half_key(planet.x, planet.y)]["planets"].append(planet)

    for fleet in fleets:
        quadrant_regions[quadrant_key(fleet.x, fleet.y)]["fleets"].append(fleet)
        vertical_regions[vertical_half_key(fleet.x, fleet.y)]["fleets"].append(fleet)
        horizontal_regions[horizontal_half_key(fleet.x, fleet.y)]["fleets"].append(fleet)

    return {
        "quadrant_regions": quadrant_regions,
        "vertical_regions": vertical_regions,
        "horizontal_regions": horizontal_regions,
    }


def position_region_key(planet, region_type):
    if region_type == "quadrant":
        return quadrant_key(planet.x, planet.y)
    if region_type == "vertical":
        return vertical_half_key(planet.x, planet.y)
    if region_type == "horizontal":
        return horizontal_half_key(planet.x, planet.y)
    raise ValueError(f"Unknown region type: {region_type}")


def region_keys(region_type):
    if region_type == "quadrant":
        return QUADRANT_KEYS
    if region_type == "vertical":
        return VERTICAL_HALF_KEYS
    if region_type == "horizontal":
        return HORIZONTAL_HALF_KEYS
    raise ValueError(f"Unknown region type: {region_type}")


def region_center(region_key):
    centers = {
        "top_left": (25.0, 25.0),
        "top_right": (75.0, 25.0),
        "bottom_left": (25.0, 75.0),
        "bottom_right": (75.0, 75.0),
        "left": (25.0, 50.0),
        "right": (75.0, 50.0),
        "top": (50.0, 25.0),
        "bottom": (50.0, 75.0),
    }
    return centers[region_key]


def planet_crosses_region_slice(planet, region_type):
    crosses_vertical = abs(planet.x - CENTER) <= planet.radius
    crosses_horizontal = abs(planet.y - CENTER) <= planet.radius
    if region_type == "quadrant":
        return crosses_vertical or crosses_horizontal
    if region_type == "vertical":
        return crosses_vertical
    if region_type == "horizontal":
        return crosses_horizontal
    return False


def nearest_region_keys(planet, keys):
    return sorted(
        keys,
        key=lambda key: distance_xy(planet.x, planet.y, *region_center(key)),
    )


def balanced_planet_regions(planets, region_type, anchor_planet=None):
    if region_type == "quadrant":
        return compact_seeded_quadrants(planets, anchor_planet=anchor_planet)

    keys = region_keys(region_type)
    regions = {key: [] for key in keys}
    target_size = len(planets) // len(keys) if keys else 0
    extra_slots = len(planets) % len(keys)
    capacities = {
        key: target_size + (1 if index < extra_slots else 0)
        for index, key in enumerate(keys)
    }

    ordered_planets = sorted(
        planets,
        key=lambda planet: (
            planet_crosses_region_slice(planet, region_type),
            planet.id,
        ),
    )

    def place(planet, candidates):
        for key in candidates:
            if len(regions[key]) < capacities[key]:
                regions[key].append(planet)
                return
        fallback = min(keys, key=lambda key: len(regions[key]))
        regions[fallback].append(planet)

    for planet in ordered_planets:
        preferred = position_region_key(planet, region_type)
        candidates = [preferred] + [
            key for key in nearest_region_keys(planet, keys) if key != preferred
        ]
        place(planet, candidates)

    for key in keys:
        regions[key].sort(key=lambda planet: planet.id)
    return regions


def angle_around_center(planet):
    angle = math.atan2(planet.y - CENTER, planet.x - CENTER)
    return angle if angle >= 0 else angle + 2 * math.pi


def balanced_angular_quadrants(planets, anchor_planet=None):
    regions = {key: [] for key in QUADRANT_KEYS}
    if not planets:
        return regions

    ordered = sorted(planets, key=lambda planet: (angle_around_center(planet), planet.id))
    base_size = len(ordered) // len(QUADRANT_KEYS)
    extra = len(ordered) % len(QUADRANT_KEYS)
    sizes = [base_size + (1 if index < extra else 0) for index in range(len(QUADRANT_KEYS))]

    if anchor_planet is not None:
        anchor_index = next(
            (index for index, planet in enumerate(ordered) if planet.id == anchor_planet.id),
            0,
        )
        first_group_size = sizes[0]
        start_index = (anchor_index - first_group_size // 2) % len(ordered)
        ordered = ordered[start_index:] + ordered[:start_index]

    cursor = 0
    groups = []
    for size in sizes:
        groups.append(ordered[cursor : cursor + size])
        cursor += size

    groups = refine_quadrant_compactness(groups, anchor_planet=anchor_planet)

    anchor_group_index = None
    key_by_centroid = []
    for group in groups:
        if anchor_planet is not None and any(planet.id == anchor_planet.id for planet in group):
            anchor_group_index = len(key_by_centroid)
        if group:
            avg_x = sum(planet.x for planet in group) / len(group)
            avg_y = sum(planet.y for planet in group) / len(group)
        else:
            avg_x, avg_y = CENTER, CENTER
        key_by_centroid.append(quadrant_key(avg_x, avg_y))

    used = set()
    for index, (group, preferred_key) in enumerate(zip(groups, key_by_centroid)):
        if anchor_group_index is not None and index == anchor_group_index:
            preferred_key = quadrant_key(anchor_planet.x, anchor_planet.y)
        key = preferred_key
        if key in used:
            key = next(candidate for candidate in QUADRANT_KEYS if candidate not in used)
        used.add(key)
        regions[key] = sorted(group, key=lambda planet: planet.id)

    return regions


def compact_seeded_quadrants(planets, anchor_planet=None):
    regions = {key: [] for key in QUADRANT_KEYS}
    if not planets:
        return regions
    if anchor_planet is None:
        return balanced_angular_quadrants(planets, anchor_planet=None)

    seeds = compact_quadrant_seeds(planets, anchor_planet)
    capacities = balanced_capacities(len(planets), len(seeds))
    groups = [[seed] for seed in seeds]
    assigned = {seed.id for seed in seeds}

    remaining = [planet for planet in planets if planet.id not in assigned]
    while remaining:
        best = None
        for planet in remaining:
            adjacency = local_adjacency_count(planet, planets)
            for group_index, group in enumerate(groups):
                if len(group) >= capacities[group_index]:
                    continue
                score = compact_assignment_score(
                    planet,
                    group,
                    seeds[group_index],
                    anchor_planet,
                    group_index == 0,
                    adjacency,
                )
                candidate = (score, group_index, planet.id, planet)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, group_index, _, planet = best
        groups[group_index].append(planet)
        remaining = [p for p in remaining if p.id != planet.id]

    groups = refine_anchor_cluster(groups, anchor_planet, planets)

    key_for_group = [quadrant_key(anchor_planet.x, anchor_planet.y)]
    used = {key_for_group[0]}
    for group in groups[1:]:
        cx, cy = group_centroid(group)
        key = quadrant_key(cx, cy)
        if key in used:
            key = next(candidate for candidate in QUADRANT_KEYS if candidate not in used)
        used.add(key)
        key_for_group.append(key)

    for key, group in zip(key_for_group, groups):
        regions[key] = sorted(group, key=lambda planet: planet.id)
    return regions


def balanced_capacities(total_items, group_count):
    base = total_items // group_count
    extra = total_items % group_count
    return [base + (1 if index < extra else 0) for index in range(group_count)]


def compact_quadrant_seeds(planets, anchor_planet):
    anchor_angle = angle_around_center(anchor_planet)
    target_angles = [
        anchor_angle,
        (anchor_angle + math.pi) % (2 * math.pi),
        (anchor_angle + math.pi / 2) % (2 * math.pi),
        (anchor_angle - math.pi / 2) % (2 * math.pi),
    ]
    seeds = [anchor_planet]
    used = {anchor_planet.id}
    for target_angle in target_angles[1:]:
        candidates = [planet for planet in planets if planet.id not in used]
        seed = min(
            candidates,
            key=lambda planet: (
                angular_distance(angle_around_center(planet), target_angle),
                distance_xy(planet.x, planet.y, CENTER, CENTER),
                planet.id,
            ),
        )
        seeds.append(seed)
        used.add(seed.id)
    return seeds


def angular_distance(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def group_centroid(group):
    if not group:
        return CENTER, CENTER
    return (
        sum(planet.x for planet in group) / len(group),
        sum(planet.y for planet in group) / len(group),
    )


def compact_assignment_score(planet, group, seed, anchor_planet, is_anchor_group, adjacency):
    cx, cy = group_centroid(group)
    score = distance_xy(planet.x, planet.y, cx, cy)
    score += 0.35 * distance(planet, seed)
    score -= 1.5 * adjacency
    if is_anchor_group:
        score += 0.45 * distance(planet, anchor_planet)
        if adjacency == 0 and planet.id != anchor_planet.id:
            score += 18.0
    return score


def local_adjacency_count(planet, planets, radius=24.0):
    return sum(
        1
        for other in planets
        if other.id != planet.id and distance(planet, other) <= radius
    )


def refine_anchor_cluster(groups, anchor_planet, planets):
    anchor_group = groups[0]
    if not anchor_group:
        return groups

    removable = sorted(
        [planet for planet in anchor_group if planet.id != anchor_planet.id],
        key=lambda planet: distance(planet, anchor_planet),
        reverse=True,
    )
    outside = sorted(
        [
            (group_index, planet)
            for group_index, group in enumerate(groups[1:], start=1)
            for planet in group
        ],
        key=lambda item: distance(item[1], anchor_planet),
    )

    for group_index, outside_planet in outside[:6]:
        if not removable:
            break
        if local_adjacency_count(outside_planet, planets) == 0:
            continue
        inside_planet = removable[0]
        before = group_compactness(groups[0])
        after_anchor_group = [
            outside_planet if planet.id == inside_planet.id else planet
            for planet in groups[0]
        ]
        after = group_compactness(after_anchor_group)
        if after < before:
            groups[0] = after_anchor_group
            groups[group_index] = [
                inside_planet if planet.id == outside_planet.id else planet
                for planet in groups[group_index]
            ]
            removable.pop(0)

    return groups


def refine_quadrant_compactness(groups, anchor_planet=None, passes=2, boundary_width=2):
    if len(groups) <= 1:
        return groups

    anchor_id = anchor_planet.id if anchor_planet is not None else None
    groups = [list(group) for group in groups]

    for _ in range(passes):
        improved = False
        for index in range(len(groups)):
            next_index = (index + 1) % len(groups)
            left = groups[index]
            right = groups[next_index]
            if not left or not right:
                continue

            left_candidates = left[-boundary_width:]
            right_candidates = right[:boundary_width]
            best_swap = None
            best_delta = 0.0
            before = group_compactness(left) + group_compactness(right)

            for left_planet in left_candidates:
                if left_planet.id == anchor_id:
                    continue
                for right_planet in right_candidates:
                    if right_planet.id == anchor_id:
                        continue
                    new_left = [right_planet if p.id == left_planet.id else p for p in left]
                    new_right = [left_planet if p.id == right_planet.id else p for p in right]
                    after = group_compactness(new_left) + group_compactness(new_right)
                    delta = before - after
                    if delta > best_delta + 1e-9:
                        best_delta = delta
                        best_swap = (left_planet, right_planet, new_left, new_right)

            if best_swap is not None:
                _, _, groups[index], groups[next_index] = best_swap
                groups[index].sort(key=lambda planet: angle_around_center(planet))
                groups[next_index].sort(key=lambda planet: angle_around_center(planet))
                improved = True

        if not improved:
            break

    return groups


def group_compactness(group):
    if len(group) <= 1:
        return 0.0
    center_x = sum(planet.x for planet in group) / len(group)
    center_y = sum(planet.y for planet in group) / len(group)
    return sum(distance_xy(planet.x, planet.y, center_x, center_y) for planet in group)


def region_key_for_planet(planet_id, regions):
    for key, planets in regions.items():
        if any(planet.id == planet_id for planet in planets):
            return key
    return None


def player_start_planet(context):
    start_planets = context["initial_planets"] or context["planets"]
    for planet in start_planets:
        if planet.owner == context["player"]:
            return planet
    for planet in context["my_planets"]:
        return planet
    return None


def future_position_from_initial(planet, turns, angular_velocity):
    if is_static_planet(planet):
        return planet.x, planet.y
    radius = orbit_radius(planet)
    initial_angle = math.atan2(planet.y - CENTER, planet.x - CENTER)
    future_angle = initial_angle + angular_velocity * turns
    return CENTER + radius * math.cos(future_angle), CENTER + radius * math.sin(future_angle)


def position_from_initial_at(planet, absolute_step, context):
    return future_position_from_initial(planet, absolute_step, context["angular_velocity"])


def fleet_steps_between_positions(source, target, ships, source_pos, target_pos):
    direct_distance = distance_xy(source_pos[0], source_pos[1], target_pos[0], target_pos[1])
    travel_distance = max(0.0, direct_distance - source.radius - target.radius)
    return max(1, math.ceil(travel_distance / fleet_speed(ships)))


def exact_intercept_from_initial(source, target, ships, launch_step, context, max_flight_steps=80):
    source_pos = position_from_initial_at(source, launch_step, context)
    for flight_steps in range(1, max_flight_steps + 1):
        arrival_step = launch_step + flight_steps
        target_pos = position_from_initial_at(target, arrival_step, context)
        angle = math.atan2(target_pos[1] - source_pos[1], target_pos[0] - source_pos[0])
        hit_step = simulate_initial_intercept_hit(
            source,
            target,
            ships,
            angle,
            launch_step,
            context,
            max_steps=flight_steps,
        )
        if hit_step == flight_steps:
            return {
                "angle": angle,
                "fleet_steps": flight_steps,
                "source_position": source_pos,
                "target_position": target_pos,
                "safe": True,
            }
    return None


def simulate_initial_intercept_hit(source, target, ships, angle, launch_step, context, max_steps):
    speed = fleet_speed(ships)
    source_pos = position_from_initial_at(source, launch_step, context)
    fleet_x = source_pos[0] + math.cos(angle) * (source.radius + 0.1)
    fleet_y = source_pos[1] + math.sin(angle) * (source.radius + 0.1)
    prev_x, prev_y = fleet_x, fleet_y
    for step in range(1, max_steps + 1):
        fleet_x += math.cos(angle) * speed
        fleet_y += math.sin(angle) * speed
        if fleet_x < 0 or fleet_x > BOARD_SIZE or fleet_y < 0 or fleet_y > BOARD_SIZE:
            return None
        if segment_hits_sun(prev_x, prev_y, fleet_x, fleet_y):
            return None
        target_pos = position_from_initial_at(target, launch_step + step, context)
        if distance_xy(fleet_x, fleet_y, target_pos[0], target_pos[1]) <= target.radius:
            return step
        prev_x, prev_y = fleet_x, fleet_y
    return None


def start_reach_profile(source_planets, target, context, horizon=10):
    cost = capture_cost(target)
    profile = []
    for player_step in range(horizon):
        target_pos = future_position_from_initial(target, player_step, context["angular_velocity"])
        best = None
        for source in source_planets:
            intercept = exact_intercept_from_initial(source, target, cost, player_step, context)
            waiting_steps = waiting_steps_to_afford(source, target, player_step)
            candidate = {
                "player_step": player_step,
                "source_id": source.id,
                "fleet_steps": intercept["fleet_steps"] if intercept else None,
                "safe": bool(intercept),
                "waiting_steps": waiting_steps,
                "source_position": intercept["source_position"] if intercept else position_from_initial_at(source, player_step, context),
                "target_position": intercept["target_position"] if intercept else target_pos,
                "angle": intercept["angle"] if intercept else None,
            }
            if best is None or (
                -candidate["waiting_steps"],
                candidate["safe"],
                -(candidate["fleet_steps"] or 999),
            ) > (
                -best["waiting_steps"],
                best["safe"],
                -(best["fleet_steps"] or 999),
            ):
                best = candidate
        profile.append(
            best
            or {
                "player_step": player_step,
                "source_id": None,
                "fleet_steps": None,
                "safe": False,
                "waiting_steps": 999,
                "source_position": None,
                "target_position": target_pos,
            }
        )
    return profile


def waiting_steps_to_afford(source, target, player_step):
    cost = capture_cost(target)
    future_source_ships = int(source.ships) + max(0, int(player_step)) * int(source.production)
    production = max(1, int(source.production))
    missing = cost - future_source_ships + 1
    if missing <= 0:
        return 0
    return math.ceil(missing / production)


def start_planet_efficiency(target, reach_profile):
    if not reach_profile:
        return 999
    return reach_profile[0]["waiting_steps"]


def start_planet_value_profile(target, reach_profile):
    cost = capture_cost(target)
    revenue = int(target.production)
    profile = []
    for item in reach_profile:
        fleet_steps = item["fleet_steps"] if item else None
        waiting_steps = item["waiting_steps"] if item else 999
        profile.append(
            {
                "player_step": item["player_step"] if item else None,
                "source_id": item["source_id"] if item else None,
                "cost": cost,
                "revenue": revenue,
                "target_production": revenue,
                "fleet_steps": fleet_steps,
                "waiting_steps": waiting_steps,
                "safe": bool(item and item["safe"]),
                "effectiveness": waiting_steps,
            }
        )
    return profile


def summarize_region(region_planets, source_planets, context):
    ranked = []
    total_cost = 0
    total_revenue = 0
    for planet in region_planets:
        total_revenue += int(planet.production)
        if planet.owner == context["player"]:
            continue
        cost = capture_cost(planet)
        total_cost += cost
        reach_profile = start_reach_profile(source_planets, planet, context)
        value_profile = start_planet_value_profile(planet, reach_profile)
        ranked.append(
            {
                "planet_id": planet.id,
                "owner": planet.owner,
                "cost": cost,
                "revenue": int(planet.production),
                "target_production": int(planet.production),
                "efficiency": start_planet_efficiency(planet, reach_profile),
                "value_by_player_step": value_profile,
                "reach_by_player_step": reach_profile,
            }
        )
    ranked.sort(
        key=lambda item: (
            item["efficiency"],
            -item["target_production"],
            item["value_by_player_step"][0]["fleet_steps"]
            if item["value_by_player_step"] and item["value_by_player_step"][0]["fleet_steps"] is not None
            else 999,
            item["planet_id"],
        )
    )
    top_targets_by_player_step = top_region_targets_by_player_step(ranked)
    return {
        "planet_ids": [planet.id for planet in region_planets],
        "planet_count": len(region_planets),
        "total_capture_cost": total_cost,
        "total_revenue": total_revenue,
        "ranked_planets": ranked,
        "top_targets_by_player_step": top_targets_by_player_step,
        "priority_target": max(
            top_targets_by_player_step,
            key=lambda item: item["priority_value"],
            default=None,
        ),
    }


def top_region_targets_by_player_step(ranked_planets, horizon=10, base_step=20):
    top_targets = []
    for step_index in range(horizon):
        candidates = []
        for planet in ranked_planets:
            value_steps = planet.get("value_by_player_step", [])
            if step_index >= len(value_steps):
                continue
            step_data = value_steps[step_index]
            effectiveness = step_data["effectiveness"]
            target_production = step_data["target_production"]
            candidates.append(
                {
                    "planet_id": planet["planet_id"],
                    "n_step": step_index + 1,
                    "effectiveness": effectiveness,
                    "target_production": target_production,
                }
            )
        if not candidates:
            continue

        top = min(
            candidates,
            key=lambda item: (
                item["effectiveness"],
                -item["target_production"],
                item["planet_id"],
            ),
        )
        top["priority_value"] = (
            base_step - top["n_step"] - top["effectiveness"]
        ) * top["target_production"]
        top_targets.append(top)
    return top_targets


def build_start_region_value_report(context):
    base_planets = context["initial_planets"] or context["planets"]
    current_by_id = context["planet_by_id"]
    start_planets = [
        base._replace(
            owner=current_by_id.get(base.id, base).owner,
            ships=current_by_id.get(base.id, base).ships,
            production=current_by_id.get(base.id, base).production,
        )
        for base in base_planets
    ]
    start_planet = next(
        (planet for planet in start_planets if planet.owner == context["player"]),
        player_start_planet(context),
    )
    source_planets = [planet for planet in start_planets if planet.owner == context["player"]]
    if not source_planets:
        source_planets = context["my_planets"]
    reports = {}
    for report_name, region_type in (
        ("quadrants", "quadrant"),
        ("vertical_halves", "vertical"),
        ("horizontal_halves", "horizontal"),
    ):
        region_map = balanced_planet_regions(
            start_planets,
            region_type,
            anchor_planet=start_planet if region_type == "quadrant" else None,
        )
        my_region_key = (
            region_key_for_planet(start_planet.id, region_map) if start_planet else None
        )
        my_region_planets = region_map.get(my_region_key, []) if my_region_key else []
        reports[report_name] = {
            "my_region_key": my_region_key,
            "my_region": summarize_region(my_region_planets, source_planets, context),
            "region_counts": {key: len(planets) for key, planets in region_map.items()},
            "all_region_planet_ids": {
                key: [planet.id for planet in planets]
                for key, planets in region_map.items()
            },
        }
        if region_type == "quadrant":
            reports[report_name]["planet_type_groups"] = build_planet_type_groups(start_planets)
            reports[report_name]["priority_groups"] = build_quadrant_priority_groups(
                region_map,
                my_region_key,
                start_planet,
            )
    return reports


def get_opening_groups(context):
    signature = opening_cache_signature(context)
    cached = OPENING_GROUP_CACHE.get(signature)
    if cached is not None:
        return cached
    groups = build_opening_groups(context)
    OPENING_GROUP_CACHE.clear()
    OPENING_GROUP_CACHE[signature] = groups
    return groups


def opening_cache_signature(context):
    base_planets = context["initial_planets"] or context["planets"]
    planet_signature = tuple(
        sorted(
            (
                planet.id,
                round(planet.x, 4),
                round(planet.y, 4),
                round(planet.radius, 4),
                int(planet.production),
            )
            for planet in base_planets
        )
    )
    return (context["player"], planet_signature)


def build_opening_groups(context):
    base_planets = context["initial_planets"] or context["planets"]
    current_by_id = context["planet_by_id"]
    start_planets = [
        base._replace(
            owner=current_by_id.get(base.id, base).owner,
            ships=current_by_id.get(base.id, base).ships,
            production=current_by_id.get(base.id, base).production,
        )
        for base in base_planets
    ]
    start_planet = next(
        (planet for planet in start_planets if planet.owner == context["player"]),
        player_start_planet(context),
    )
    source_planets = [planet for planet in start_planets if planet.owner == context["player"]]
    if not source_planets:
        source_planets = context["my_planets"]

    region_map = balanced_planet_regions(
        start_planets,
        "quadrant",
        anchor_planet=start_planet,
    )
    my_region_key = region_key_for_planet(start_planet.id, region_map) if start_planet else None
    my_region_planets = region_map.get(my_region_key, []) if my_region_key else []
    priority_target = summarize_region(
        my_region_planets,
        source_planets,
        context,
    ).get("priority_target")
    return {
        "priority_groups": build_quadrant_priority_groups(
            region_map,
            my_region_key,
            start_planet,
        ),
        "rotating_groups": build_rotating_priority_groups(
            region_map,
            my_region_key,
            context["angular_velocity"],
        ),
        "priority_target": priority_target,
    }


def build_planet_type_groups(planets):
    static_planets = [planet for planet in planets if is_static_planet(planet)]
    rotating_planets = [planet for planet in planets if not is_static_planet(planet)]
    return {
        "static_planet_ids": [planet.id for planet in static_planets],
        "rotating_planet_ids": [planet.id for planet in rotating_planets],
        "static_production": sum(int(planet.production) for planet in static_planets),
        "rotating_production": sum(int(planet.production) for planet in rotating_planets),
    }


def build_quadrant_priority_groups(region_map, my_region_key, anchor_planet):
    if my_region_key is None:
        return []

    neighbors = adjacent_quadrants(my_region_key)
    opposite = opposite_quadrant(my_region_key)
    rotating_anchor = select_rotating_transit_planet(region_map.get(my_region_key, []), anchor_planet)
    second_region = select_rotating_direction_neighbor(
        neighbors,
        rotating_anchor,
        anchor_planet,
        region_map,
    )
    third_region = next((key for key in neighbors if key != second_region), None)

    groups = [
        {
            "rank": 1,
            "role": "primary_base_region",
            "region_key": my_region_key,
        },
        {
            "rank": 2,
            "role": "rotating_direction_neighbor",
            "region_key": second_region,
        },
        {
            "rank": 3,
            "role": "other_neighbor",
            "region_key": third_region,
        },
        {
            "rank": 4,
            "role": "opposite_region",
            "region_key": opposite,
        },
    ]

    for group in groups:
        region_key = group["region_key"]
        region_planets = region_map.get(region_key, [])
        group["planet_ids"] = [planet.id for planet in region_planets if is_static_planet(planet)]
        del group["region_key"]
    return groups


def build_rotating_priority_groups(region_map, my_region_key, angular_velocity):
    if my_region_key is None:
        return []

    neighbors = adjacent_quadrants(my_region_key)
    opposite = opposite_quadrant(my_region_key)
    second_region = against_rotation_neighbor(my_region_key, neighbors, angular_velocity)
    third_region = next((key for key in neighbors if key != second_region), None)
    group_specs = [
        (1, "initial_rotating_region", my_region_key),
        (2, "against_rotation_neighbor", second_region),
        (3, "other_rotating_neighbor", third_region),
        (4, "opposite_rotating_region", opposite),
    ]
    groups = []
    for rank, role, region_key in group_specs:
        region_planets = region_map.get(region_key, [])
        groups.append(
            {
                "rank": rank,
                "role": role,
                "planet_ids": [
                    planet.id for planet in region_planets if not is_static_planet(planet)
                ],
            }
        )
    return groups


def against_rotation_neighbor(region_key, neighbors, angular_velocity):
    if not neighbors:
        return None
    if abs(angular_velocity) <= 1e-12:
        return neighbors[0]

    base_angle = region_center_angle(region_key)
    desired_sign = -1 if angular_velocity > 0 else 1

    def score(candidate):
        delta = normalize_angle(region_center_angle(candidate) - base_angle)
        signed_match = 1 if delta * desired_sign > 0 else 0
        return (signed_match, -abs(abs(delta) - math.pi / 2), candidate)

    return max(neighbors, key=score)


def region_center_angle(region_key):
    x, y = region_center(region_key)
    return math.atan2(y - CENTER, x - CENTER)


def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def adjacent_quadrants(region_key):
    adjacent = {
        "top_left": ["top_right", "bottom_left"],
        "top_right": ["top_left", "bottom_right"],
        "bottom_left": ["top_left", "bottom_right"],
        "bottom_right": ["top_right", "bottom_left"],
    }
    return adjacent.get(region_key, [])


def opposite_quadrant(region_key):
    opposite = {
        "top_left": "bottom_right",
        "top_right": "bottom_left",
        "bottom_left": "top_right",
        "bottom_right": "top_left",
    }
    return opposite.get(region_key)


def select_rotating_transit_planet(planets, anchor_planet):
    rotating = [planet for planet in planets if not is_static_planet(planet)]
    if not rotating:
        return None
    if anchor_planet is None:
        return max(rotating, key=lambda planet: (planet.production, -planet.ships, -planet.id))
    return min(
        rotating,
        key=lambda planet: (
            distance(planet, anchor_planet),
            -planet.production,
            planet.ships,
            planet.id,
        ),
    )


def select_rotating_direction_neighbor(neighbors, rotating_anchor, anchor_planet, region_map):
    if not neighbors:
        return None
    if rotating_anchor is None or anchor_planet is None:
        typed = build_region_planet_types(region_map)
        return max(
            neighbors,
            key=lambda key: (
                typed[key]["static_production"],
                typed[key]["total_production"],
                key,
            ),
        )

    vx = rotating_anchor.x - anchor_planet.x
    vy = rotating_anchor.y - anchor_planet.y
    mag = math.hypot(vx, vy)
    if mag <= 1e-9:
        return neighbors[0]
    vx /= mag
    vy /= mag

    def neighbor_alignment(region_key):
        cx, cy = region_center(region_key)
        nx = cx - anchor_planet.x
        ny = cy - anchor_planet.y
        nmag = math.hypot(nx, ny)
        if nmag <= 1e-9:
            return -2.0
        return (vx * nx + vy * ny) / nmag

    return max(neighbors, key=lambda key: (neighbor_alignment(key), key))


def build_region_planet_types(region_map):
    result = {}
    for region_key, planets in region_map.items():
        static_planets = [planet for planet in planets if is_static_planet(planet)]
        result[region_key] = {
            "static_production": sum(int(planet.production) for planet in static_planets),
            "total_production": sum(int(planet.production) for planet in planets),
        }
    return result


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def distance_xy(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def score_by_player(planets, fleets):
    scores = defaultdict(int)
    for planet in planets:
        if planet.owner != -1:
            scores[planet.owner] += int(planet.ships)
    for fleet in fleets:
        scores[fleet.owner] += int(fleet.ships)
    return dict(scores)


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(max(1, ships)) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


def travel_time(source, target, ships):
    travel_distance = max(0.0, distance(source, target) - source.radius - target.radius)
    return max(1, math.ceil(travel_distance / fleet_speed(ships)))


def capture_cost(target, margin=1):
    return max(1, int(target.ships) + margin)


def future_garrison(planet, turns):
    ships = int(planet.ships)
    if planet.owner != -1:
        ships += int(planet.production) * max(0, int(turns))
    return ships


def target_value(target, turns, context):
    remaining_after_arrival = max(0, context["remaining_steps"] - turns)
    production_value = target.production * remaining_after_arrival
    owner_bonus = 25 if target.owner not in (-1, context["player"]) else 0
    comet_penalty = 20 if target.id in context["comet_ids"] and comet_remaining_life(target.id, context) < turns + 12 else 0
    return production_value + owner_bonus - target.ships - comet_penalty


def orbit_radius(planet):
    return distance_xy(planet.x, planet.y, CENTER, CENTER)


def is_static_planet(planet):
    return orbit_radius(planet) + planet.radius >= ROTATION_RADIUS_LIMIT


def is_orbiting_planet(planet, context):
    return planet.id not in context["comet_ids"] and not is_static_planet(planet)


def future_planet_position(planet, turns, context):
    if planet.id in context["comet_ids"]:
        return future_comet_position(planet.id, turns, context)

    initial = context["initial_by_id"].get(planet.id)
    if initial is None or is_static_planet(initial):
        return planet.x, planet.y

    radius = orbit_radius(initial)
    current_angle = math.atan2(planet.y - CENTER, planet.x - CENTER)
    future_angle = current_angle + context["angular_velocity"] * turns
    return CENTER + radius * math.cos(future_angle), CENTER + radius * math.sin(future_angle)


def future_comet_position(planet_id, turns, context):
    for group in context["comets"]:
        planet_ids = group.get("planet_ids", [])
        if planet_id not in planet_ids:
            continue
        path_index = int(group.get("path_index", 0) or 0) + int(turns)
        path = group.get("paths", [])[planet_ids.index(planet_id)]
        if 0 <= path_index < len(path):
            return path[path_index][0], path[path_index][1]
        return None
    return None


def comet_remaining_life(planet_id, context):
    for group in context["comets"]:
        planet_ids = group.get("planet_ids", [])
        if planet_id not in planet_ids:
            continue
        path = group.get("paths", [])[planet_ids.index(planet_id)]
        return max(0, len(path) - int(group.get("path_index", 0) or 0))
    return 0


def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return distance_xy(px, py, ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return distance_xy(px, py, ax + t * dx, ay + t * dy)


def segment_hits_sun(ax, ay, bx, by, safety=0.0):
    return point_to_segment_distance(CENTER, CENTER, ax, ay, bx, by) <= SUN_RADIUS + safety


def shot_angle(source, target_x, target_y):
    return math.atan2(target_y - source.y, target_x - source.x)


def is_safe_shot(source, target_x, target_y, safety=0.5):
    angle = shot_angle(source, target_x, target_y)
    start_x = source.x + math.cos(angle) * (source.radius + 0.1)
    start_y = source.y + math.sin(angle) * (source.radius + 0.1)
    return not segment_hits_sun(start_x, start_y, target_x, target_y, safety=safety)


def aim_at_future_position(source, target, ships, context, max_iterations=4):
    turns = travel_time(source, target, ships)
    target_pos = (target.x, target.y)
    for _ in range(max_iterations):
        future_pos = future_planet_position(target, turns, context)
        if future_pos is None:
            return None
        target_pos = future_pos
        proxy = target._replace(x=target_pos[0], y=target_pos[1])
        next_turns = travel_time(source, proxy, ships)
        if abs(next_turns - turns) <= 1:
            break
        turns = next_turns
    if not is_safe_shot(source, target_pos[0], target_pos[1]):
        return None
    return shot_angle(source, target_pos[0], target_pos[1]), turns, target_pos


def exact_intercept_now(source, target, ships, context, max_flight_steps=80):
    for flight_steps in range(1, max_flight_steps + 1):
        target_pos = future_planet_position(target, flight_steps, context)
        if target_pos is None:
            continue
        angle = math.atan2(target_pos[1] - source.y, target_pos[0] - source.x)
        hit_step = simulate_live_intercept_hit(
            source,
            target,
            ships,
            angle,
            context,
            max_steps=flight_steps,
        )
        if hit_step == flight_steps:
            return angle, flight_steps, target_pos
    return None


def simulate_live_intercept_hit(source, target, ships, angle, context, max_steps):
    speed = fleet_speed(ships)
    fleet_x = source.x + math.cos(angle) * (source.radius + 0.1)
    fleet_y = source.y + math.sin(angle) * (source.radius + 0.1)
    prev_x, prev_y = fleet_x, fleet_y
    for step in range(1, max_steps + 1):
        fleet_x += math.cos(angle) * speed
        fleet_y += math.sin(angle) * speed
        if fleet_x < 0 or fleet_x > BOARD_SIZE or fleet_y < 0 or fleet_y > BOARD_SIZE:
            return None
        if segment_hits_sun(prev_x, prev_y, fleet_x, fleet_y):
            return None
        target_pos = future_planet_position(target, step, context)
        if target_pos is None:
            return None
        if distance_xy(fleet_x, fleet_y, target_pos[0], target_pos[1]) <= target.radius:
            return step
        prev_x, prev_y = fleet_x, fleet_y
    return None


def fleet_position_after(fleet, turns):
    speed = fleet_speed(fleet.ships)
    distance_traveled = speed * turns
    return (
        fleet.x + math.cos(fleet.angle) * distance_traveled,
        fleet.y + math.sin(fleet.angle) * distance_traveled,
    )


def fleet_hits_planet(fleet, planet, max_turns=120):
    for turn in range(1, max_turns + 1):
        x, y = fleet_position_after(fleet, turn)
        if distance_xy(x, y, planet.x, planet.y) <= planet.radius:
            return turn
        if x < 0 or x > BOARD_SIZE or y < 0 or y > BOARD_SIZE:
            return None
    return None


def incoming_fleets_by_planet(context, max_turns=80):
    incoming = defaultdict(list)
    for fleet in context["fleets"]:
        for planet in context["planets"]:
            eta = fleet_hits_planet(fleet, planet, max_turns=max_turns)
            if eta is not None:
                incoming[planet.id].append(
                    {"eta": eta, "owner": fleet.owner, "ships": fleet.ships, "fleet_id": fleet.id}
                )
                break
    return dict(incoming)


def threatened_planets(context):
    incoming = incoming_fleets_by_planet(context)
    threatened = []
    for planet in context["my_planets"]:
        enemy_power = 0
        soonest = None
        for item in incoming.get(planet.id, []):
            if item["owner"] == context["player"]:
                continue
            enemy_power += item["ships"]
            soonest = item["eta"] if soonest is None else min(soonest, item["eta"])
        if soonest is not None and enemy_power > future_garrison(planet, soonest):
            threatened.append({"planet": planet, "eta": soonest, "enemy_power": enemy_power})
    return threatened


def choose_moves(context):
    moves = []
    target = quadrant_priority_target(context)
    if target is None:
        return moves

    needed = capture_cost(target)
    best_source = None
    best_aim = None
    for source in sorted(context["my_planets"], key=lambda p: (-p.ships, p.id)):
        if source.ships <= needed:
            continue
        aim = exact_intercept_now(source, target, needed, context)
        if aim is None:
            continue
        if best_source is None or aim[1] < best_aim[1]:
            best_source = source
            best_aim = aim

    if best_source is not None:
        moves.append([best_source.id, best_aim[0], needed])
    return moves


def quadrant_priority_target(context):
    quadrant_report = context["start_region_value_report"]["quadrants"]["my_region"]
    priority = quadrant_report.get("priority_target")
    if priority is None:
        return None
    target = context["planet_by_id"].get(priority["planet_id"])
    if target is None or target.owner == context["player"]:
        return None
    return target


def agent(obs, config=None):
    context = build_context(obs)
    return choose_moves(context)
