import math
from collections import defaultdict


BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
MAX_STEPS = 500
STAT_HORIZON = 12
FLOW_HORIZON = 20
FLOW_TARGET_LIMIT = 8

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


def _agent_impl(obs, config=None):
    ctx = build_context(obs)
    if not ctx["my_idx"]:
        return []

    candidates = generate_unified_flow_candidates(ctx)
    candidates = [cand for cand in candidates if cand[C_SCORE] > ctx["policy"]["min_score"] - 25]
    candidates.sort(key=lambda cand: cand[C_SCORE], reverse=True)
    candidates = validate_top_candidates(candidates, ctx, valid_limit=48, scan_limit=160)
    return select_unified_flow_moves(candidates, ctx)


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
    build_statistical_state(ctx)
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


def build_statistical_state(ctx):
    planet_count = len(ctx["p_id"])
    safety_floor = [0] * planet_count
    minimum_margin = [0] * planet_count
    reinforcement_demand = [0] * planet_count
    donor_capacity = [0] * planet_count
    pressure_eta = [0] * planet_count

    for planet_i in ctx["my_idx"]:
        floor = 3 + ctx["p_prod"][planet_i] * 2 + (3 if ctx["step"] < 35 else 0)
        safety_floor[planet_i] = floor
        balance = ctx["p_ships"][planet_i]
        min_margin = balance - floor
        last_eta = 0
        has_enemy_pressure = False
        breach_eta = 0
        for event in ctx["incoming"].get(planet_i, []):
            eta = event["eta"]
            if eta > STAT_HORIZON:
                break
            balance += ctx["p_prod"][planet_i] * max(0, eta - last_eta)
            if event["owner"] == ctx["player"]:
                balance += event["ships"]
            else:
                balance -= event["ships"]
                has_enemy_pressure = True
            margin = balance - floor
            if margin < min_margin:
                min_margin = margin
                if margin < 0 and breach_eta == 0:
                    breach_eta = eta
            last_eta = eta

        minimum_margin[planet_i] = min_margin
        demand = max(0, -min_margin) if has_enemy_pressure else 0
        reinforcement_demand[planet_i] = demand
        pressure_eta[planet_i] = breach_eta
        reserve_available = max(0, min_margin)
        policy_cap = int(ctx["p_ships"][planet_i] * ctx["policy"]["max_source_commit_fraction"])
        donor_capacity[planet_i] = min(reserve_available, policy_cap)

    reach_weight = [[0.0] * planet_count for _ in range(planet_count)]
    friendly_support_influence = [0.0] * planet_count
    best_friendly_eta = [STAT_HORIZON + 1] * planet_count
    for source_i in ctx["my_idx"]:
        available = donor_capacity[source_i]
        if available <= 0:
            continue
        probe_ships = max(1, available)
        for target_i in range(planet_count):
            if target_i == source_i:
                continue
            eta = cheap_travel_time_i(source_i, target_i, probe_ships, ctx)
            weight = 1.0 / (1.0 + eta * eta)
            reach_weight[source_i][target_i] = weight
            friendly_support_influence[target_i] += available * weight
            if eta < best_friendly_eta[target_i]:
                best_friendly_eta[target_i] = eta

    exposure = [0.0] * planet_count
    front_relevance = [0.0] * planet_count
    for planet_i in ctx["my_idx"]:
        exposure[planet_i] = reinforcement_demand[planet_i] / (
            1.0 + friendly_support_influence[planet_i]
        )
    for target_i in ctx["enemy_idx"] + ctx["neutral_idx"]:
        if ctx["p_owner"][target_i] == -1:
            opportunity = max(0.0, ctx["neutral_intrinsic_score"][target_i])
        else:
            opportunity = max(0.0, ctx["p_prod"][target_i] * 5.0 + 20.0 - (ctx["p_ships"][target_i] + 1))
        eta = best_friendly_eta[target_i]
        resistance = max(1.0, ctx["p_ships"][target_i] + ctx["p_prod"][target_i] * eta)
        feasibility = min(1.0, friendly_support_influence[target_i] / resistance)
        front_relevance[target_i] = opportunity * feasibility

    support_hub_score = [0.0] * planet_count
    for source_i in ctx["my_idx"]:
        if donor_capacity[source_i] <= 0:
            continue
        opportunity_reach = 0.0
        for target_i in ctx["enemy_idx"] + ctx["neutral_idx"]:
            opportunity_reach += reach_weight[source_i][target_i] * front_relevance[target_i]
        safety_factor = 1.0 / (1.0 + exposure[source_i])
        support_hub_score[source_i] = donor_capacity[source_i] * opportunity_reach * safety_factor

    ctx["safety_floor"] = safety_floor
    ctx["minimum_margin"] = minimum_margin
    ctx["reinforcement_demand"] = reinforcement_demand
    ctx["donor_capacity"] = donor_capacity
    ctx["pressure_eta"] = pressure_eta
    ctx["friendly_reach_weight"] = reach_weight
    ctx["friendly_support_influence"] = friendly_support_influence
    ctx["exposure"] = exposure
    ctx["front_relevance"] = front_relevance
    ctx["support_hub_score"] = support_hub_score


def generate_unified_flow_candidates(ctx):
    candidates = []
    sources = usable_sources(ctx)
    if not sources:
        return candidates

    friendly_demands = sorted(
        (
            i
            for i in ctx["my_idx"]
            if ctx["reinforcement_demand"][i] > 0
        ),
        key=lambda i: (
            ctx["pressure_eta"][i] or STAT_HORIZON + 1,
            -ctx["exposure"][i],
            -ctx["reinforcement_demand"][i],
        ),
    )
    pressure_targets = sorted(
        ctx["enemy_idx"] + ctx["neutral_idx"],
        key=lambda i: (
            -ctx["front_relevance"][i],
            -target_intrinsic_value(i, ctx),
            ctx["p_id"][i],
        ),
    )[:FLOW_TARGET_LIMIT]

    for source_i in sources:
        for target_i in friendly_demands[:8]:
            if source_i == target_i:
                continue
            cand = build_support_flow_candidate(source_i, target_i, ctx)
            if cand is not None:
                candidates.append(cand)

        local_targets = sorted(
            pressure_targets,
            key=lambda i: (
                -source_target_attention(source_i, i, ctx),
                ctx["p_id"][i],
            ),
        )[:5]
        for target_i in local_targets:
            cand = build_pressure_flow_candidate(source_i, target_i, ctx)
            if cand is not None:
                candidates.append(cand)

    return candidates


def target_intrinsic_value(target_i, ctx):
    owner = ctx["p_owner"][target_i]
    capture_cost = ctx["p_ships"][target_i] + 1
    if owner == -1:
        return ctx["p_prod"][target_i] * 12.0 - capture_cost
    if owner != ctx["player"]:
        return ctx["p_prod"][target_i] * 5.0 + 20.0 - capture_cost
    return ctx["reinforcement_demand"][target_i] * 4.0 + ctx["p_prod"][target_i] * 2.0


def source_target_attention(source_i, target_i, ctx):
    return (
        target_intrinsic_value(target_i, ctx)
        + ctx["front_relevance"][target_i]
        - distance_i(source_i, target_i, ctx) * 0.35
    )


def build_support_flow_candidate(source_i, target_i, ctx):
    available = source_available_ships(source_i, ctx)
    need = ctx["reinforcement_demand"][target_i]
    if available <= 0 or need <= 0:
        return None

    ships = min(available, need + ctx["p_prod"][target_i])
    eta = estimate_arrival(source_i, target_i, ships, ctx)
    pressure_eta = ctx["pressure_eta"][target_i] or STAT_HORIZON
    if eta > pressure_eta + 3:
        return None

    fulfilled = min(ships, need)
    urgency = max(0, STAT_HORIZON + 2 - pressure_eta) * 2.5
    score = (
        fulfilled * 4.0
        + ctx["p_prod"][target_i] * 6.0
        + urgency
        + ctx["exposure"][target_i] * 8.0
        - eta * 0.8
        - ships * ctx["policy"]["commitment_cost_rate"]
        - ctx["policy"]["action_tax"] * 0.5
    )
    angle = aim_angle(source_i, target_i, eta, ctx)
    return make_candidate(source_i, target_i, ships, angle, eta, score, "FLOW")


def build_pressure_flow_candidate(source_i, target_i, ctx):
    available = source_available_ships(source_i, ctx)
    if available <= 0:
        return None

    projected_eta = estimate_arrival(source_i, target_i, max(1, available), ctx)
    _, projected_ships = cheap_forecast_target(target_i, projected_eta, ctx)
    required = max(1, int(projected_ships) + 1)
    first_sizes = unique_clamped(
        [
            required,
            int(required * 0.6),
            available,
        ],
        1,
        available,
    )

    best = None
    for first_ships in first_sizes:
        for delay in (0, 4):
            result = simulate_friendly_wave_plan(source_i, target_i, first_ships, delay, ctx)
            if result is None:
                continue
            score, eta = result
            if best is None or score > best[0]:
                best = (score, first_ships, eta)
    if best is None:
        return None

    score, ships, eta = best
    angle = aim_angle(source_i, target_i, eta, ctx)
    return make_candidate(source_i, target_i, ships, angle, eta, score, "FLOW")


def simulate_friendly_wave_plan(source_i, target_i, first_ships, second_delay, ctx):
    if first_ships <= 0 or first_ships > source_available_ships(source_i, ctx):
        return None

    first_eta = estimate_arrival(source_i, target_i, first_ships, ctx)
    if ctx["step"] + first_eta >= MAX_STEPS:
        return None

    waves = [(first_eta, first_ships)]
    second_ships = 0
    if second_delay > 0:
        future_source_ships = (
            ctx["p_ships"][source_i]
            - first_ships
            + ctx["p_prod"][source_i] * second_delay
        )
        second_ships = max(
            0,
            min(
                int(first_ships * 0.65),
                future_source_ships - ctx["safety_floor"][source_i],
            ),
        )
        if second_ships > 0:
            second_eta = second_delay + estimate_arrival(source_i, target_i, second_ships, ctx)
            waves.append((second_eta, second_ships))

    horizon = min(
        ctx["remaining_steps"],
        max(FLOW_HORIZON, max(arrival for arrival, _ in waves) + 6),
    )
    owner = ctx["p_owner"][target_i]
    original_owner = owner
    ships = ctx["p_ships"][target_i]
    known_by_turn = defaultdict(list)
    for event in ctx["incoming"].get(target_i, []):
        if event["eta"] <= horizon:
            known_by_turn[event["eta"]].append((event["owner"], event["ships"]))
    for arrival, wave_ships in waves:
        if arrival <= horizon:
            known_by_turn[arrival].append((ctx["player"], wave_ships))

    captured = False
    capture_turn = 0
    enemy_denial_turns = 0
    our_hold_turns = 0
    for turn in range(1, horizon + 1):
        if owner != -1:
            ships += ctx["p_prod"][target_i]
        for event_owner, event_ships in known_by_turn.get(turn, ()):
            owner, ships = resolve_arrival(owner, ships, event_owner, event_ships)
        if owner == ctx["player"]:
            our_hold_turns += 1
            if not captured:
                captured = True
                capture_turn = turn
        if original_owner not in (-1, ctx["player"]) and owner != original_owner:
            enemy_denial_turns += 1

    first_margin = ctx["minimum_margin"][source_i] - first_ships
    source_risk = max(0, -first_margin) * 3.0
    capture_bonus = 28.0 if captured else 0.0
    temporary_pressure = min(first_ships, ctx["p_ships"][target_i]) * 0.3
    production_value = our_hold_turns * ctx["p_prod"][target_i]
    denial_value = enemy_denial_turns * ctx["p_prod"][target_i] * 1.25
    timing_value = max(0, 25 - (capture_turn or first_eta)) * 0.4
    future_wave_cost = second_ships * 0.08
    planned_power = first_ships + second_ships
    projected_resistance = max(
        1,
        ctx["p_ships"][target_i] + ctx["p_prod"][target_i] * first_eta,
    )
    pressure_ratio = planned_power / projected_resistance
    if not captured and pressure_ratio < 0.45:
        return None
    comet_penalty = 0.0
    target_id = ctx["p_id"][target_i]
    if target_id in ctx["comet_ids"] and comet_remaining_life(target_id, ctx) <= first_eta + 8:
        comet_penalty = 45.0

    score = (
        target_intrinsic_value(target_i, ctx)
        + ctx["front_relevance"][target_i]
        + capture_bonus
        + temporary_pressure
        + production_value
        + denial_value
        + timing_value
        - first_ships * ctx["policy"]["commitment_cost_rate"]
        - future_wave_cost
        - ctx["policy"]["action_tax"]
        - source_risk
        - comet_penalty
    )
    if not captured:
        score -= planned_power * 0.55
    return score, first_eta


def resolve_arrival(owner, ships, event_owner, event_ships):
    if event_owner == owner:
        return owner, ships + event_ships
    if event_ships > ships:
        return event_owner, event_ships - ships
    return owner, ships - event_ships


def usable_sources(ctx):
    return sorted(
        (i for i in ctx["my_idx"] if ctx["donor_capacity"][i] > 0),
        key=lambda i: (
            -ctx["donor_capacity"][i],
            -ctx["support_hub_score"][i],
            -ctx["p_prod"][i],
            ctx["p_id"][i],
        ),
    )


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


def select_unified_flow_moves(candidates, ctx):
    policy = ctx["policy"]
    moves = []
    budget = {i: source_available_ships(i, ctx) for i in ctx["my_idx"]}
    allocated = defaultdict(int)
    selected_edges = set()
    total_committed = 0

    for _ in range(policy["max_new_fleets"]):
        best = None
        best_score = policy["min_score"]
        for cand in candidates:
            source_i = cand[C_SOURCE_I]
            target_i = cand[C_TARGET_I]
            ships = cand[C_SHIPS]
            edge = (source_i, target_i)
            if edge in selected_edges or ships > budget.get(source_i, 0):
                continue
            if total_committed + ships > policy["max_total_commit"]:
                continue

            if ctx["p_owner"][target_i] == ctx["player"]:
                gap = max(0, ctx["reinforcement_demand"][target_i] - allocated[target_i])
                fulfilled = min(ships, gap)
                excess = max(0, ships - gap)
                marginal = fulfilled * 3.5 - excess * 0.7
            else:
                _, projected_ships = cheap_forecast_target(target_i, cand[C_ETA], ctx)
                capture_gap = max(1, int(projected_ships) + 1 - allocated[target_i])
                contribution = min(ships, capture_gap)
                crossing_bonus = 32.0 if ships >= capture_gap else 0.0
                excess = max(0, ships - capture_gap)
                marginal = contribution * 0.45 + crossing_bonus - excess * 0.15

            score = cand[C_SCORE] + marginal
            if score > best_score:
                best = cand
                best_score = score

        if best is None:
            break

        source_i = best[C_SOURCE_I]
        target_i = best[C_TARGET_I]
        ships = best[C_SHIPS]
        moves.append([ctx["p_id"][source_i], best[C_ANGLE], ships])
        budget[source_i] -= ships
        allocated[target_i] += ships
        total_committed += ships
        selected_edges.add((source_i, target_i))

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


def source_available_ships(source_i, ctx):
    return ctx["donor_capacity"][source_i]


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


def agent(obs, config=None):
    return _agent_impl(obs, config)
