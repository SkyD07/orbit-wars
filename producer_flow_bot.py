import math
from collections import defaultdict, namedtuple


BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
MAX_STEPS = 500

Planet = namedtuple("Planet", "id owner x y radius ships production")
Fleet = namedtuple("Fleet", "id owner x y angle from_planet_id ships")
Candidate = namedtuple("Candidate", "source_id target_id ships angle eta score mission")


def read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def agent(obs, config=None):
    ctx = build_context(obs)
    if not ctx["my_planets"]:
        return []

    candidates = []
    candidates.extend(gen_defense_candidates(ctx))
    candidates.extend(gen_snipe_candidates(ctx))
    candidates.extend(gen_capture_candidates(ctx))
    candidates.extend(gen_attack_candidates(ctx))
    candidates.extend(gen_endgame_candidates(ctx))

    candidates = [cand for cand in candidates if cand.score > ctx["policy"]["min_score"] - 25]
    candidates.sort(key=lambda cand: cand.score, reverse=True)
    candidates = validate_top_candidates(candidates, ctx, valid_limit=80, scan_limit=350)
    return select_moves_greedy(candidates, ctx)


def build_context(obs):
    player = int(read(obs, "player", 0) or 0)
    step = int(read(obs, "step", 0) or 0)
    planets = [Planet(*p) for p in (read(obs, "planets", []) or [])]
    fleets = [Fleet(*f) for f in (read(obs, "fleets", []) or [])]
    initial_planets = [Planet(*p) for p in (read(obs, "initial_planets", []) or [])]
    comet_ids = set(read(obs, "comet_planet_ids", []) or [])
    ctx = {
        "player": player,
        "step": step,
        "remaining_steps": MAX_STEPS - step,
        "planets": planets,
        "planet_by_id": {p.id: p for p in planets},
        "fleets": fleets,
        "initial_by_id": {p.id: p for p in initial_planets},
        "comets": read(obs, "comets", []) or [],
        "comet_ids": comet_ids,
        "angular_velocity": float(read(obs, "angular_velocity", 0.0) or 0.0),
    }
    ctx["my_planets"] = [p for p in planets if p.owner == player]
    ctx["enemy_planets"] = [p for p in planets if p.owner not in (-1, player)]
    ctx["neutral_planets"] = [p for p in planets if p.owner == -1]
    ctx["enemy_fleets"] = [f for f in fleets if f.owner != player]
    ctx["incoming"] = rough_incoming_by_target(ctx)
    ctx["policy"] = build_strategy_policy(ctx)
    return ctx


def build_strategy_policy(ctx):
    my_planet_ships = sum(int(p.ships) for p in ctx["my_planets"])
    my_fleet_ships = sum(int(f.ships) for f in ctx["fleets"] if f.owner == ctx["player"])
    owned_stock = my_planet_ships + my_fleet_ships
    return {
        "owned_stock": owned_stock,
        "max_new_fleets": 6,
        "min_score": 8.0,
        "action_tax": 7.0,
        "commitment_cost_rate": 0.18,
        "max_source_commit_fraction": 0.55,
        "max_total_commit_fraction": 0.35,
        "max_total_commit": max(12, int(owned_stock * 0.35)),
    }


def gen_capture_candidates(ctx):
    candidates = []
    for source in usable_sources(ctx):
        targets = candidate_targets_for_source(source, ctx["neutral_planets"], global_limit=14)
        for target in targets:
            for ships in ship_buckets(source, target, ctx):
                cand = score_move(source, target, ships, ctx, "CAPTURE_NEUTRAL")
                if cand:
                    candidates.append(cand)
    return candidates


def gen_attack_candidates(ctx):
    candidates = []
    for source in usable_sources(ctx):
        targets = candidate_targets_for_source(source, ctx["enemy_planets"], global_limit=12)
        for target in targets:
            for ships in ship_buckets(source, target, ctx):
                cand = score_move(source, target, ships, ctx, "ATTACK_ENEMY_PRODUCER")
                if cand:
                    candidates.append(cand)
    return candidates


def gen_snipe_candidates(ctx):
    candidates = []
    opportunities = {}
    for fleet in ctx["enemy_fleets"]:
        hit = nearest_future_hit(fleet, ctx, max_turns=55)
        if hit is None:
            continue
        target, enemy_eta = hit
        if target.owner == ctx["player"]:
            continue
        before_owner, before_ships = cheap_forecast_target(target, max(0, enemy_eta - 1), ctx)
        after_owner, after_ships = cheap_forecast_target(target, enemy_eta, ctx)
        if after_owner == before_owner and after_ships > max(3, target.production * 2):
            continue
        current = opportunities.get(target.id)
        if current is None or enemy_eta < current:
            opportunities[target.id] = enemy_eta
    for target_id, enemy_eta in opportunities.items():
        target = ctx["planet_by_id"].get(target_id)
        if target is None:
            continue
        for source in usable_sources(ctx):
            for ships in ship_buckets(source, target, ctx):
                cand = score_move(source, target, ships, ctx, "SNIPE_CONTESTED")
                if not cand:
                    continue
                if cand.eta < enemy_eta + 1 or cand.eta > enemy_eta + 4:
                    continue
                candidates.append(cand._replace(score=cand.score + 18 + max(0, 5 - abs(cand.eta - enemy_eta - 2)) * 3))
    return candidates


def gen_defense_candidates(ctx):
    candidates = []
    threats = threatened_planets(ctx)
    if not threats:
        return candidates
    for target, need, eta in threats[:8]:
        for source in usable_sources(ctx):
            if source.id == target.id:
                continue
            buckets = unique_clamped(
                [need, need + target.production * 2, int(source.ships * 0.35)],
                1,
                source_available_ships(source, ctx),
            )
            for ships in buckets:
                arrival = estimate_arrival(source, target, ships, ctx)
                if arrival > eta + 3:
                    continue
                angle = aim_angle(source, target, arrival, ctx)
                policy = ctx["policy"]
                score = (
                    60
                    + need * 0.8
                    + target.production * max(0, ctx["remaining_steps"] - arrival) * 0.25
                    - ships * policy["commitment_cost_rate"]
                    - policy["action_tax"] * 0.5
                )
                candidates.append(Candidate(source.id, target.id, ships, angle, arrival, score, "REINFORCE_THREATENED"))
    return candidates


def gen_endgame_candidates(ctx):
    if ctx["remaining_steps"] > 45:
        return []
    candidates = []
    for source in usable_sources(ctx):
        available = int(source.ships) - reserve(source, ctx)
        if available < 3:
            continue
        targets = sorted(ctx["enemy_planets"] + ctx["neutral_planets"], key=lambda p: (distance(source, p), p.ships))[:6]
        for target in targets:
            ships = min(available, max(1, int(target.ships) + 1))
            cand = score_move(source, target, ships, ctx, "ENDGAME_SCORE_DUMP")
            if cand:
                candidates.append(cand._replace(score=cand.score + ships * 0.25))
    return candidates


def usable_sources(ctx):
    return sorted(
        (p for p in ctx["my_planets"] if int(p.ships) > reserve(p, ctx) + 1),
        key=lambda p: (-p.ships, -p.production, p.id),
    )


def candidate_targets_for_source(source, targets, global_limit):
    if not targets:
        return []
    selected = []
    seen = set()

    def add(items):
        for target in items:
            if target.id in seen:
                continue
            seen.add(target.id)
            selected.append(target)

    add(sorted(targets, key=lambda p: (-p.production, p.ships, distance_to_center(p), p.id))[:global_limit])
    add(sorted(targets, key=lambda p: (distance(source, p), p.ships, -p.production, p.id))[:5])
    add(
        sorted(
            targets,
            key=lambda p: (
                (int(p.ships) + 1) / max(1, p.production),
                distance(source, p),
                p.id,
            ),
        )[:5]
    )
    return selected


def ship_buckets(source, target, ctx):
    arrival0 = estimate_arrival(source, target, max(1, int(target.ships) + 1), ctx)
    predicted_owner, predicted_ships = cheap_forecast_target(target, arrival0, ctx)
    base = max(1, int(predicted_ships) + (1 if predicted_owner != ctx["player"] else 0))
    available = source_available_ships(source, ctx)
    return unique_clamped(
        [
            base,
            base + int(target.production) * 2,
            int(base * 1.15) + 1,
            int(base * 1.35) + 1,
            int(source.ships * 0.25),
            int(source.ships * 0.50),
            int(source.ships * 0.75),
        ],
        1,
        available,
    )


def score_move(source, target, ships, ctx, mission):
    policy = ctx["policy"]
    if ships <= 0 or ships > source_available_ships(source, ctx):
        return None
    eta = estimate_arrival(source, target, ships, ctx)
    if eta <= 0 or ctx["step"] + eta >= MAX_STEPS:
        return None
    angle = aim_angle(source, target, eta, ctx)
    target_owner, target_ships = cheap_forecast_target(target, eta, ctx)
    margin = ships - target_ships
    remaining = max(0, ctx["remaining_steps"] - eta)

    if target_owner != ctx["player"] and margin <= 0 and mission != "REINFORCE_THREATENED":
        return None

    if target_owner == ctx["player"]:
        gain = ships * 0.15 + target.production * min(eta, 8)
    elif margin > 0:
        production_gain = target.production * remaining
        denial = target.production * remaining * 0.8 if target_owner != -1 else 0.0
        gain = production_gain + denial + margin * 0.2
    else:
        gain = -ships * 0.8

    timing_bonus = max(0, 25 - eta) * 0.45
    source_penalty = source_safety_penalty(source, ships, ctx)
    late_penalty = max(0, ctx["step"] + eta - 430) * 0.5
    comet_penalty = 45 if target.id in ctx["comet_ids"] and comet_remaining_life(target.id, ctx) <= eta + 8 else 0
    enemy_bonus = 20 if mission == "ATTACK_ENEMY_PRODUCER" and target.owner not in (-1, ctx["player"]) else 0
    action_tax = policy["action_tax"] * (0.65 if mission == "SNIPE_CONTESTED" else 1.0)
    commitment_cost = ships * policy["commitment_cost_rate"]
    score = gain + timing_bonus + enemy_bonus - commitment_cost - action_tax - source_penalty - late_penalty - comet_penalty
    return Candidate(source.id, target.id, int(ships), angle, eta, score, mission)


def validate_top_candidates(candidates, ctx, valid_limit=80, scan_limit=350):
    valid = []
    for cand in candidates[:scan_limit]:
        source = ctx["planet_by_id"].get(cand.source_id)
        target = ctx["planet_by_id"].get(cand.target_id)
        if source is None or target is None:
            continue
        if source.owner != ctx["player"]:
            continue
        if not path_is_reasonably_safe(source, target, cand.angle, cand.ships, cand.eta, ctx):
            continue
        valid.append(cand)
        if len(valid) >= valid_limit:
            break
    return valid


def select_moves_greedy(candidates, ctx):
    policy = ctx["policy"]
    moves = []
    budget = {p.id: source_available_ships(p, ctx) for p in ctx["my_planets"]}
    target_pressure = defaultdict(int)
    selected_intents = set()
    total_committed = 0
    for _ in range(policy["max_new_fleets"]):
        best = None
        best_score = policy["min_score"]
        for cand in candidates:
            intent = (cand.source_id, cand.target_id, cand.mission)
            if intent in selected_intents or cand.ships > budget.get(cand.source_id, 0):
                continue
            if total_committed + cand.ships > policy["max_total_commit"]:
                continue
            target = ctx["planet_by_id"].get(cand.target_id)
            if target is None:
                continue
            pressure_penalty = max(0, target_pressure[cand.target_id] - int(target.ships)) * 0.65
            score = cand.score - pressure_penalty
            if score > best_score:
                best = cand
                best_score = score
        if best is None:
            break
        moves.append([best.source_id, best.angle, best.ships])
        budget[best.source_id] -= best.ships
        total_committed += best.ships
        target_pressure[best.target_id] += best.ships
        selected_intents.add((best.source_id, best.target_id, best.mission))
    return moves


def cheap_forecast_target(target, turns, ctx):
    owner = target.owner
    ships = int(target.ships)
    last_t = 0
    arrivals = [item for item in ctx["incoming"].get(target.id, []) if item["eta"] <= turns]
    arrivals.sort(key=lambda item: item["eta"])
    for fleet in arrivals:
        dt = fleet["eta"] - last_t
        if owner != -1:
            ships += int(target.production) * max(0, dt)
        if fleet["owner"] == owner:
            ships += fleet["ships"]
        elif fleet["ships"] > ships:
            owner = fleet["owner"]
            ships = fleet["ships"] - ships
        else:
            ships -= fleet["ships"]
        last_t = fleet["eta"]
    if owner != -1:
        ships += int(target.production) * max(0, turns - last_t)
    return owner, max(0, ships)


def rough_incoming_by_target(ctx):
    incoming = defaultdict(list)
    for fleet in ctx["fleets"]:
        hit = nearest_future_hit(fleet, ctx, max_turns=70)
        if hit is not None:
            target, eta = hit
            incoming[target.id].append({"eta": eta, "owner": fleet.owner, "ships": int(fleet.ships)})
    for items in incoming.values():
        items.sort(key=lambda item: item["eta"])
    return incoming


def nearest_future_hit(fleet, ctx, max_turns=70):
    speed = fleet_speed(fleet.ships)
    x = fleet.x
    y = fleet.y
    for turn in range(1, max_turns + 1):
        prev_x, prev_y = x, y
        x += math.cos(fleet.angle) * speed
        y += math.sin(fleet.angle) * speed
        if x < 0 or x > BOARD_SIZE or y < 0 or y > BOARD_SIZE:
            return None
        if point_segment_distance(CENTER, CENTER, prev_x, prev_y, x, y) <= SUN_RADIUS:
            return None
        first_hit = first_planet_collision(prev_x, prev_y, x, y, turn, ctx)
        if first_hit is not None:
            return first_hit, turn
    return None


def threatened_planets(ctx):
    threats = []
    for planet in ctx["my_planets"]:
        enemy_power = 0
        soonest = None
        for fleet in ctx["incoming"].get(planet.id, []):
            if fleet["owner"] == ctx["player"]:
                continue
            enemy_power += fleet["ships"]
            soonest = fleet["eta"] if soonest is None else min(soonest, fleet["eta"])
        if soonest is None:
            continue
        projected = int(planet.ships) + int(planet.production) * soonest
        if enemy_power > projected:
            threats.append((planet, enemy_power - projected + 2, soonest))
    threats.sort(key=lambda item: (item[2], -item[1]))
    return threats


def estimate_arrival(source, target, ships, ctx):
    turns = cheap_travel_time(source.x, source.y, source.radius, target, ships)
    for _ in range(2):
        pos = future_planet_position(target, turns, ctx)
        if pos is None:
            break
        proxy = target._replace(x=pos[0], y=pos[1])
        next_turns = cheap_travel_time(source.x, source.y, source.radius, proxy, ships)
        if abs(next_turns - turns) <= 1:
            turns = next_turns
            break
        turns = next_turns
    return max(1, turns)


def cheap_travel_time(sx, sy, sr, target, ships):
    travel = max(0.0, math.hypot(target.x - sx, target.y - sy) - sr - target.radius)
    return max(1, int(math.ceil(travel / fleet_speed(ships))))


def aim_angle(source, target, eta, ctx):
    pos = future_planet_position(target, eta, ctx)
    if pos is None:
        pos = (target.x, target.y)
    return math.atan2(pos[1] - source.y, pos[0] - source.x)


def path_is_reasonably_safe(source, target, angle, ships, eta, ctx):
    speed = fleet_speed(ships)
    x = source.x + math.cos(angle) * (source.radius + 0.1)
    y = source.y + math.sin(angle) * (source.radius + 0.1)
    prev_x, prev_y = x, y
    for turn in range(1, min(eta + 2, 90) + 1):
        x += math.cos(angle) * speed
        y += math.sin(angle) * speed
        if x < 0 or x > BOARD_SIZE or y < 0 or y > BOARD_SIZE:
            return False
        if point_segment_distance(CENTER, CENTER, prev_x, prev_y, x, y) <= SUN_RADIUS + 0.25:
            return False
        first_hit = first_planet_collision(prev_x, prev_y, x, y, turn, ctx)
        if first_hit is not None:
            return first_hit.id == target.id
        prev_x, prev_y = x, y
    return False


def first_planet_collision(ax, ay, bx, by, turn, ctx):
    best_planet = None
    best_t = None
    for planet in ctx["planets"]:
        pos = future_planet_position(planet, turn, ctx)
        if pos is None:
            continue
        t = segment_circle_hit_fraction(pos[0], pos[1], planet.radius, ax, ay, bx, by)
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t = t
            best_planet = planet
    return best_planet


def segment_circle_hit_fraction(cx, cy, radius, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    fx = ax - cx
    fy = ay - cy
    a = dx * dx + dy * dy
    if a <= 1e-12:
        return 0.0 if math.hypot(ax - cx, ay - cy) <= radius else None
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    hits = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
    if not hits:
        return None
    return min(hits)


def source_safety_penalty(source, ships, ctx):
    remaining = int(source.ships) - ships
    reserve_ships = reserve(source, ctx)
    if remaining >= reserve_ships:
        return 0.0
    return (reserve_ships - remaining) * 2.2


def source_available_ships(source, ctx):
    reserve_available = max(0, int(source.ships) - reserve(source, ctx))
    policy_cap = int(source.ships * ctx["policy"]["max_source_commit_fraction"])
    return max(0, min(reserve_available, policy_cap))


def reserve(source, ctx):
    base = 3 + int(source.production) * 2
    if ctx["step"] < 35:
        base += 3
    for item in ctx["incoming"].get(source.id, []):
        if item["owner"] != ctx["player"] and item["eta"] <= 12:
            base += item["ships"]
    return min(int(source.ships), base)


def unique_clamped(values, low, high):
    if high < low:
        return []
    result = []
    seen = set()
    for value in values:
        value = int(value)
        value = max(low, min(high, value))
        if value not in seen:
            seen.add(value)
            result.append(value)
    return sorted(result)


def future_planet_position(planet, turns, ctx):
    if planet.id in ctx["comet_ids"]:
        return future_comet_position(planet.id, turns, ctx)
    initial = ctx["initial_by_id"].get(planet.id)
    if initial is None or is_static_planet(initial):
        return planet.x, planet.y
    radius = math.hypot(initial.x - CENTER, initial.y - CENTER)
    current_angle = math.atan2(planet.y - CENTER, planet.x - CENTER)
    future_angle = current_angle + ctx["angular_velocity"] * turns
    return CENTER + radius * math.cos(future_angle), CENTER + radius * math.sin(future_angle)


def future_comet_position(planet_id, turns, ctx):
    for group in ctx["comets"]:
        planet_ids = group.get("planet_ids", [])
        if planet_id not in planet_ids:
            continue
        index = planet_ids.index(planet_id)
        path = group.get("paths", [])[index]
        path_index = int(group.get("path_index", 0) or 0) + int(turns)
        if 0 <= path_index < len(path):
            return path[path_index][0], path[path_index][1]
        return None
    return None


def comet_remaining_life(planet_id, ctx):
    for group in ctx["comets"]:
        planet_ids = group.get("planet_ids", [])
        if planet_id not in planet_ids:
            continue
        path = group.get("paths", [])[planet_ids.index(planet_id)]
        return max(0, len(path) - int(group.get("path_index", 0) or 0))
    return 0


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(max(1, ships)) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


def is_static_planet(planet):
    return math.hypot(planet.x - CENTER, planet.y - CENTER) + planet.radius >= 50.0


def point_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def distance_to_center(planet):
    return math.hypot(planet.x - CENTER, planet.y - CENTER)
