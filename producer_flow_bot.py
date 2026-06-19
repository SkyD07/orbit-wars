import math
from collections import defaultdict


BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
MAX_STEPS = 500

P_ID = 0
P_OWNER = 1
P_X = 2
P_Y = 3
P_RADIUS = 4
P_SHIPS = 5
P_PROD = 6

F_ID = 0
F_OWNER = 1
F_X = 2
F_Y = 3
F_ANGLE = 4
F_FROM = 5
F_SHIPS = 6

XY_X = 0
XY_Y = 1

C_SOURCE_I = 0
C_TARGET_I = 1
C_SHIPS = 2
C_ANGLE = 3
C_ETA = 4
C_SCORE = 5
C_MISSION = 6

def read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def make_candidate(source_i, target_i, ships, angle, eta, score, mission):
    return (source_i, target_i, int(ships), angle, eta, score, mission)


def candidate_with_score(candidate, score):
    return (
        candidate[C_SOURCE_I],
        candidate[C_TARGET_I],
        candidate[C_SHIPS],
        candidate[C_ANGLE],
        candidate[C_ETA],
        score,
        candidate[C_MISSION],
    )


def agent(obs, config=None):
    ctx = build_context(obs)
    if not ctx["my_idx"]:
        return []

    candidates = []
    candidates.extend(gen_defense_candidates(ctx))
    candidates.extend(gen_snipe_candidates(ctx))
    candidates.extend(gen_capture_candidates(ctx))
    candidates.extend(gen_attack_candidates(ctx))
    candidates.extend(gen_endgame_candidates(ctx))

    candidates = [cand for cand in candidates if cand[C_SCORE] > ctx["policy"]["min_score"] - 25]
    candidates.sort(key=lambda cand: cand[C_SCORE], reverse=True)
    candidates = validate_top_candidates(candidates, ctx, valid_limit=80, scan_limit=350)
    return select_moves_greedy(candidates, ctx)


def build_context(obs):
    player = int(read(obs, "player", 0) or 0)
    step = int(read(obs, "step", 0) or 0)
    raw_planets = read(obs, "planets", []) or []
    raw_fleets = read(obs, "fleets", []) or []
    raw_initial_planets = read(obs, "initial_planets", []) or []
    comet_ids = set(read(obs, "comet_planet_ids", []) or [])
    p_id = [int(p[P_ID]) for p in raw_planets]
    p_owner = [int(p[P_OWNER]) for p in raw_planets]
    p_x = [float(p[P_X]) for p in raw_planets]
    p_y = [float(p[P_Y]) for p in raw_planets]
    p_radius = [float(p[P_RADIUS]) for p in raw_planets]
    p_ships = [int(p[P_SHIPS]) for p in raw_planets]
    p_prod = [int(p[P_PROD]) for p in raw_planets]
    f_id = [int(f[F_ID]) for f in raw_fleets]
    f_owner = [int(f[F_OWNER]) for f in raw_fleets]
    f_x = [float(f[F_X]) for f in raw_fleets]
    f_y = [float(f[F_Y]) for f in raw_fleets]
    f_angle = [float(f[F_ANGLE]) for f in raw_fleets]
    f_from = [int(f[F_FROM]) for f in raw_fleets]
    f_ships = [int(f[F_SHIPS]) for f in raw_fleets]
    initial_orbit_radius_by_id = {}
    initial_static_by_id = {}
    for p in raw_initial_planets:
        pid = int(p[P_ID])
        x = float(p[P_X])
        y = float(p[P_Y])
        radius = float(p[P_RADIUS])
        initial_orbit_radius_by_id[pid] = math.hypot(x - CENTER, y - CENTER)
        initial_static_by_id[pid] = is_static_geometry(x, y, radius)
    ctx = {
        "player": player,
        "step": step,
        "remaining_steps": MAX_STEPS - step,
        "p_id": p_id,
        "p_owner": p_owner,
        "p_x": p_x,
        "p_y": p_y,
        "p_radius": p_radius,
        "p_ships": p_ships,
        "p_prod": p_prod,
        "f_id": f_id,
        "f_owner": f_owner,
        "f_x": f_x,
        "f_y": f_y,
        "f_angle": f_angle,
        "f_from": f_from,
        "f_ships": f_ships,
        "initial_orbit_radius_by_id": initial_orbit_radius_by_id,
        "initial_static_by_id": initial_static_by_id,
        "comets": read(obs, "comets", []) or [],
        "comet_ids": comet_ids,
        "angular_velocity": float(read(obs, "angular_velocity", 0.0) or 0.0),
    }
    ctx["my_idx"] = [i for i, owner in enumerate(p_owner) if owner == player]
    ctx["enemy_idx"] = [i for i, owner in enumerate(p_owner) if owner not in (-1, player)]
    ctx["neutral_idx"] = [i for i, owner in enumerate(p_owner) if owner == -1]
    ctx["enemy_fleet_idx"] = [i for i, owner in enumerate(f_owner) if owner != player]
    ctx["neutral_intrinsic_score"] = [prod * 12.0 - (ships + 1) for prod, ships in zip(p_prod, p_ships)]
    ctx["incoming"] = rough_incoming_by_target(ctx)
    ctx["policy"] = build_strategy_policy(ctx)
    return ctx


def build_strategy_policy(ctx):
    my_planet_ships = sum(ctx["p_ships"][i] for i in ctx["my_idx"])
    my_fleet_ships = sum(ctx["f_ships"][i] for i, owner in enumerate(ctx["f_owner"]) if owner == ctx["player"])
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
    for source_i in usable_sources(ctx):
        targets = neutral_targets_for_source(source_i, ctx, limit=10)
        for target_i in targets:
            for ships in ship_buckets(source_i, target_i, ctx):
                cand = score_move(source_i, target_i, ships, ctx, "CAPTURE_NEUTRAL")
                if cand:
                    candidates.append(cand)
    return candidates


def gen_attack_candidates(ctx):
    candidates = []
    for source_i in usable_sources(ctx):
        targets = candidate_targets_for_source(source_i, ctx["enemy_idx"], global_limit=12, ctx=ctx)
        for target_i in targets:
            for ships in ship_buckets(source_i, target_i, ctx):
                cand = score_move(source_i, target_i, ships, ctx, "ATTACK_ENEMY_PRODUCER")
                if cand:
                    candidates.append(cand)
    return candidates


def gen_snipe_candidates(ctx):
    candidates = []
    opportunities = {}
    for fleet_i in ctx["enemy_fleet_idx"]:
        hit = nearest_future_hit(fleet_i, ctx, max_turns=55)
        if hit is None:
            continue
        target_i, enemy_eta = hit
        if ctx["p_owner"][target_i] == ctx["player"]:
            continue
        before_owner, before_ships = cheap_forecast_target(target_i, max(0, enemy_eta - 1), ctx)
        after_owner, after_ships = cheap_forecast_target(target_i, enemy_eta, ctx)
        if after_owner == before_owner and after_ships > max(3, ctx["p_prod"][target_i] * 2):
            continue
        current = opportunities.get(target_i)
        if current is None or enemy_eta < current:
            opportunities[target_i] = enemy_eta
    for target_i, enemy_eta in opportunities.items():
        for source in usable_sources(ctx):
            for ships in ship_buckets(source, target_i, ctx):
                cand = score_move(source, target_i, ships, ctx, "SNIPE_CONTESTED")
                if not cand:
                    continue
                if cand[C_ETA] < enemy_eta + 1 or cand[C_ETA] > enemy_eta + 4:
                    continue
                candidates.append(
                    candidate_with_score(
                        cand,
                        cand[C_SCORE] + 18 + max(0, 5 - abs(cand[C_ETA] - enemy_eta - 2)) * 3,
                    )
                )
    return candidates


def gen_defense_candidates(ctx):
    candidates = []
    threats = threatened_planets(ctx)
    if not threats:
        return candidates
    for target_i, need, eta in threats[:8]:
        for source_i in usable_sources(ctx):
            if source_i == target_i:
                continue
            buckets = unique_clamped(
                [need, need + ctx["p_prod"][target_i] * 2, int(ctx["p_ships"][source_i] * 0.35)],
                1,
                source_available_ships(source_i, ctx),
            )
            for ships in buckets:
                arrival = estimate_arrival(source_i, target_i, ships, ctx)
                if arrival > eta + 3:
                    continue
                angle = aim_angle(source_i, target_i, arrival, ctx)
                policy = ctx["policy"]
                score = (
                    60
                    + need * 0.8
                    + ctx["p_prod"][target_i] * max(0, ctx["remaining_steps"] - arrival) * 0.25
                    - ships * policy["commitment_cost_rate"]
                    - policy["action_tax"] * 0.5
                )
                candidates.append(make_candidate(source_i, target_i, ships, angle, arrival, score, "REINFORCE_THREATENED"))
    return candidates


def gen_endgame_candidates(ctx):
    if ctx["remaining_steps"] > 45:
        return []
    candidates = []
    for source_i in usable_sources(ctx):
        available = source_available_ships(source_i, ctx)
        if available < 3:
            continue
        targets = sorted(ctx["enemy_idx"] + ctx["neutral_idx"], key=lambda i: (distance_i(source_i, i, ctx), ctx["p_ships"][i]))[:6]
        for target_i in targets:
            ships = min(available, max(1, ctx["p_ships"][target_i] + 1))
            cand = score_move(source_i, target_i, ships, ctx, "ENDGAME_SCORE_DUMP")
            if cand:
                candidates.append(candidate_with_score(cand, cand[C_SCORE] + ships * 0.25))
    return candidates


def usable_sources(ctx):
    return sorted(
        (i for i in ctx["my_idx"] if ctx["p_ships"][i] > reserve(i, ctx) + 1),
        key=lambda i: (-ctx["p_ships"][i], -ctx["p_prod"][i], ctx["p_id"][i]),
    )


def candidate_targets_for_source(source_i, targets, global_limit, ctx):
    if not targets:
        return []
    selected = []
    seen = set()

    def add(items):
        for target_i in items:
            if target_i in seen:
                continue
            seen.add(target_i)
            selected.append(target_i)

    add(sorted(targets, key=lambda i: (-ctx["p_prod"][i], ctx["p_ships"][i], distance_to_center_i(i, ctx), ctx["p_id"][i]))[:global_limit])
    add(sorted(targets, key=lambda i: (distance_i(source_i, i, ctx), ctx["p_ships"][i], -ctx["p_prod"][i], ctx["p_id"][i]))[:5])
    add(
        sorted(
            targets,
            key=lambda i: (
                (ctx["p_ships"][i] + 1) / max(1, ctx["p_prod"][i]),
                distance_i(source_i, i, ctx),
                ctx["p_id"][i],
            ),
        )[:5]
    )
    return selected


def neutral_targets_for_source(source_i, ctx, limit=10):
    return sorted(
        ctx["neutral_idx"],
        key=lambda i: (
            -(ctx["neutral_intrinsic_score"][i] - distance_i(source_i, i, ctx) * 0.35),
            ctx["p_id"][i],
        ),
    )[:limit]


def ship_buckets(source_i, target_i, ctx):
    arrival0 = estimate_arrival(source_i, target_i, max(1, ctx["p_ships"][target_i] + 1), ctx)
    predicted_owner, predicted_ships = cheap_forecast_target(target_i, arrival0, ctx)
    base = max(1, int(predicted_ships) + (1 if predicted_owner != ctx["player"] else 0))
    available = source_available_ships(source_i, ctx)
    return unique_clamped(
        [
            base,
            base + ctx["p_prod"][target_i] * 2,
            int(base * 1.15) + 1,
            int(base * 1.35) + 1,
            int(ctx["p_ships"][source_i] * 0.25),
            int(ctx["p_ships"][source_i] * 0.50),
            int(ctx["p_ships"][source_i] * 0.75),
        ],
        1,
        available,
    )


def score_move(source_i, target_i, ships, ctx, mission):
    policy = ctx["policy"]
    if ships <= 0 or ships > source_available_ships(source_i, ctx):
        return None
    eta = estimate_arrival(source_i, target_i, ships, ctx)
    if eta <= 0 or ctx["step"] + eta >= MAX_STEPS:
        return None
    angle = aim_angle(source_i, target_i, eta, ctx)
    target_owner, target_ships = cheap_forecast_target(target_i, eta, ctx)
    margin = ships - target_ships
    remaining = max(0, ctx["remaining_steps"] - eta)

    if target_owner != ctx["player"] and margin <= 0 and mission != "REINFORCE_THREATENED":
        return None

    if target_owner == ctx["player"]:
        gain = ships * 0.15 + ctx["p_prod"][target_i] * min(eta, 8)
    elif margin > 0:
        production_gain = ctx["p_prod"][target_i] * remaining
        denial = ctx["p_prod"][target_i] * remaining * 0.8 if target_owner != -1 else 0.0
        gain = production_gain + denial + margin * 0.2
    else:
        gain = -ships * 0.8

    timing_bonus = max(0, 25 - eta) * 0.45
    source_penalty = source_safety_penalty(source_i, ships, ctx)
    late_penalty = max(0, ctx["step"] + eta - 430) * 0.5
    target_id = ctx["p_id"][target_i]
    comet_penalty = 45 if target_id in ctx["comet_ids"] and comet_remaining_life(target_id, ctx) <= eta + 8 else 0
    enemy_bonus = 20 if mission == "ATTACK_ENEMY_PRODUCER" and ctx["p_owner"][target_i] not in (-1, ctx["player"]) else 0
    action_tax = policy["action_tax"] * (0.65 if mission == "SNIPE_CONTESTED" else 1.0)
    commitment_cost = ships * policy["commitment_cost_rate"]
    score = gain + timing_bonus + enemy_bonus - commitment_cost - action_tax - source_penalty - late_penalty - comet_penalty
    return make_candidate(source_i, target_i, ships, angle, eta, score, mission)


def validate_top_candidates(candidates, ctx, valid_limit=80, scan_limit=350):
    valid = []
    for cand in candidates[:scan_limit]:
        if cand[C_SOURCE_I] >= len(ctx["p_id"]) or cand[C_TARGET_I] >= len(ctx["p_id"]):
            continue
        if ctx["p_owner"][cand[C_SOURCE_I]] != ctx["player"]:
            continue
        if not path_is_reasonably_safe(cand[C_SOURCE_I], cand[C_TARGET_I], cand[C_ANGLE], cand[C_SHIPS], cand[C_ETA], ctx):
            continue
        valid.append(cand)
        if len(valid) >= valid_limit:
            break
    return valid


def select_moves_greedy(candidates, ctx):
    policy = ctx["policy"]
    moves = []
    budget = {i: source_available_ships(i, ctx) for i in ctx["my_idx"]}
    target_pressure = defaultdict(int)
    selected_intents = set()
    total_committed = 0
    for _ in range(policy["max_new_fleets"]):
        best = None
        best_score = policy["min_score"]
        for cand in candidates:
            intent = (cand[C_SOURCE_I], cand[C_TARGET_I], cand[C_MISSION])
            if intent in selected_intents or cand[C_SHIPS] > budget.get(cand[C_SOURCE_I], 0):
                continue
            if total_committed + cand[C_SHIPS] > policy["max_total_commit"]:
                continue
            if cand[C_TARGET_I] >= len(ctx["p_ships"]):
                continue
            pressure_penalty = max(0, target_pressure[cand[C_TARGET_I]] - ctx["p_ships"][cand[C_TARGET_I]]) * 0.65
            score = cand[C_SCORE] - pressure_penalty
            if score > best_score:
                best = cand
                best_score = score
        if best is None:
            break
        moves.append([ctx["p_id"][best[C_SOURCE_I]], best[C_ANGLE], best[C_SHIPS]])
        budget[best[C_SOURCE_I]] -= best[C_SHIPS]
        total_committed += best[C_SHIPS]
        target_pressure[best[C_TARGET_I]] += best[C_SHIPS]
        selected_intents.add((best[C_SOURCE_I], best[C_TARGET_I], best[C_MISSION]))
    return moves


def cheap_forecast_target(target_i, turns, ctx):
    owner = ctx["p_owner"][target_i]
    ships = ctx["p_ships"][target_i]
    last_t = 0
    arrivals = [item for item in ctx["incoming"].get(target_i, []) if item["eta"] <= turns]
    arrivals.sort(key=lambda item: item["eta"])
    for fleet in arrivals:
        dt = fleet["eta"] - last_t
        if owner != -1:
            ships += ctx["p_prod"][target_i] * max(0, dt)
        if fleet["owner"] == owner:
            ships += fleet["ships"]
        elif fleet["ships"] > ships:
            owner = fleet["owner"]
            ships = fleet["ships"] - ships
        else:
            ships -= fleet["ships"]
        last_t = fleet["eta"]
    if owner != -1:
        ships += ctx["p_prod"][target_i] * max(0, turns - last_t)
    return owner, max(0, ships)


def rough_incoming_by_target(ctx):
    incoming = defaultdict(list)
    for fleet_i in range(len(ctx["f_owner"])):
        hit = nearest_future_hit(fleet_i, ctx, max_turns=70)
        if hit is not None:
            target_i, eta = hit
            incoming[target_i].append({"eta": eta, "owner": ctx["f_owner"][fleet_i], "ships": ctx["f_ships"][fleet_i]})
    for items in incoming.values():
        items.sort(key=lambda item: item["eta"])
    return incoming


def nearest_future_hit(fleet_i, ctx, max_turns=70):
    speed = fleet_speed(ctx["f_ships"][fleet_i])
    x = ctx["f_x"][fleet_i]
    y = ctx["f_y"][fleet_i]
    angle = ctx["f_angle"][fleet_i]
    for turn in range(1, max_turns + 1):
        prev_x, prev_y = x, y
        x += math.cos(angle) * speed
        y += math.sin(angle) * speed
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
    for planet_i in ctx["my_idx"]:
        enemy_power = 0
        soonest = None
        for fleet in ctx["incoming"].get(planet_i, []):
            if fleet["owner"] == ctx["player"]:
                continue
            enemy_power += fleet["ships"]
            soonest = fleet["eta"] if soonest is None else min(soonest, fleet["eta"])
        if soonest is None:
            continue
        projected = ctx["p_ships"][planet_i] + ctx["p_prod"][planet_i] * soonest
        if enemy_power > projected:
            threats.append((planet_i, enemy_power - projected + 2, soonest))
    threats.sort(key=lambda item: (item[2], -item[1]))
    return threats


def estimate_arrival(source_i, target_i, ships, ctx):
    turns = cheap_travel_time_i(source_i, target_i, ships, ctx)
    for _ in range(2):
        pos = future_planet_position_i(target_i, turns, ctx)
        if pos is None:
            break
        next_turns = cheap_travel_time_xy(
            ctx["p_x"][source_i],
            ctx["p_y"][source_i],
            ctx["p_radius"][source_i],
            pos[XY_X],
            pos[XY_Y],
            ctx["p_radius"][target_i],
            ships,
        )
        if abs(next_turns - turns) <= 1:
            turns = next_turns
            break
        turns = next_turns
    return max(1, turns)


def cheap_travel_time_i(source_i, target_i, ships, ctx):
    return cheap_travel_time_xy(
        ctx["p_x"][source_i],
        ctx["p_y"][source_i],
        ctx["p_radius"][source_i],
        ctx["p_x"][target_i],
        ctx["p_y"][target_i],
        ctx["p_radius"][target_i],
        ships,
    )


def cheap_travel_time_xy(sx, sy, sr, tx, ty, tr, ships):
    travel = max(0.0, math.hypot(tx - sx, ty - sy) - sr - tr)
    return max(1, int(math.ceil(travel / fleet_speed(ships))))


def aim_angle(source_i, target_i, eta, ctx):
    pos = future_planet_position_i(target_i, eta, ctx)
    if pos is None:
        pos = (ctx["p_x"][target_i], ctx["p_y"][target_i])
    return math.atan2(pos[XY_Y] - ctx["p_y"][source_i], pos[XY_X] - ctx["p_x"][source_i])


def path_is_reasonably_safe(source_i, target_i, angle, ships, eta, ctx):
    speed = fleet_speed(ships)
    x = ctx["p_x"][source_i] + math.cos(angle) * (ctx["p_radius"][source_i] + 0.1)
    y = ctx["p_y"][source_i] + math.sin(angle) * (ctx["p_radius"][source_i] + 0.1)
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
            return first_hit == target_i
        prev_x, prev_y = x, y
    return False


def first_planet_collision(ax, ay, bx, by, turn, ctx):
    best_planet_i = None
    best_t = None
    for planet_i in range(len(ctx["p_id"])):
        pos = future_planet_position_i(planet_i, turn, ctx)
        if pos is None:
            continue
        t = segment_circle_hit_fraction(pos[XY_X], pos[XY_Y], ctx["p_radius"][planet_i], ax, ay, bx, by)
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t = t
            best_planet_i = planet_i
    return best_planet_i


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


def source_safety_penalty(source_i, ships, ctx):
    remaining = ctx["p_ships"][source_i] - ships
    reserve_ships = reserve(source_i, ctx)
    if remaining >= reserve_ships:
        return 0.0
    return (reserve_ships - remaining) * 2.2


def source_available_ships(source_i, ctx):
    reserve_available = max(0, ctx["p_ships"][source_i] - reserve(source_i, ctx))
    policy_cap = int(ctx["p_ships"][source_i] * ctx["policy"]["max_source_commit_fraction"])
    return max(0, min(reserve_available, policy_cap))


def reserve(source_i, ctx):
    base = 3 + ctx["p_prod"][source_i] * 2
    if ctx["step"] < 35:
        base += 3
    for item in ctx["incoming"].get(source_i, []):
        if item["owner"] != ctx["player"] and item["eta"] <= 12:
            base += item["ships"]
    return min(ctx["p_ships"][source_i], base)


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


def future_planet_position_i(planet_i, turns, ctx):
    planet_id = ctx["p_id"][planet_i]
    if planet_id in ctx["comet_ids"]:
        return future_comet_position(planet_id, turns, ctx)
    orbit_radius = ctx["initial_orbit_radius_by_id"].get(planet_id)
    if orbit_radius is None or ctx["initial_static_by_id"].get(planet_id, True):
        return ctx["p_x"][planet_i], ctx["p_y"][planet_i]
    current_angle = math.atan2(ctx["p_y"][planet_i] - CENTER, ctx["p_x"][planet_i] - CENTER)
    future_angle = current_angle + ctx["angular_velocity"] * turns
    return CENTER + orbit_radius * math.cos(future_angle), CENTER + orbit_radius * math.sin(future_angle)


def future_comet_position(planet_id, turns, ctx):
    for group in ctx["comets"]:
        planet_ids = group.get("planet_ids", [])
        if planet_id not in planet_ids:
            continue
        index = planet_ids.index(planet_id)
        path = group.get("paths", [])[index]
        path_index = int(group.get("path_index", 0) or 0) + int(turns)
        if 0 <= path_index < len(path):
            return path[path_index][XY_X], path[path_index][XY_Y]
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


def is_static_geometry(x, y, radius):
    return math.hypot(x - CENTER, y - CENTER) + radius >= 50.0


def point_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_i(a_i, b_i, ctx):
    return math.hypot(ctx["p_x"][a_i] - ctx["p_x"][b_i], ctx["p_y"][a_i] - ctx["p_y"][b_i])


def distance_to_center_i(planet_i, ctx):
    return math.hypot(ctx["p_x"][planet_i] - CENTER, ctx["p_y"][planet_i] - CENTER)
