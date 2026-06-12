import importlib.util
import inspect
import math
import random
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars import orbit_wars


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"

app = Flask(__name__, static_folder=str(STATIC_DIR))

env = None
bot_agent = None
medium_agent = None
bot_name = "medium"
public_agents = {}
public_modules = {}
action_history = []
current_match_mode = "4p"

PUBLIC_BOTS = {
    "public-roman-1224": ROOT / "public_bots" / "roman-lb-1224" / "notebook.py",
    "public-zachary-1000": ROOT
    / "public_bots"
    / "zachary-1000"
    / "orbit-wars-heuristic-agent-scored-1000.py",
    "public-marco-1060": ROOT
    / "public_bots"
    / "marco-1060"
    / "marco-dg-v3-3-top-score-1060-5.py",
}

FOUR_PLAYER_LINEUP = [
    ("public-roman-1224", "Public 1224"),
    ("public-zachary-1000", "Public 1000"),
    ("public-marco-1060", "Public 1060"),
    ("starter", "Starter"),
]

TWO_PLAYER_LINEUP = [
    ("public-roman-1224", "Public 1224"),
    ("public-marco-1060", "Public 1060"),
]

DEV_TWO_PLAYER_LINEUP = [
    ("producer_flow_bot.py", "ProducerFlowBot"),
    ("public-marco-1060", "Public 1060"),
]

LINEUPS = {
    "4p": FOUR_PLAYER_LINEUP,
    "2p": TWO_PLAYER_LINEUP,
    "dev2p": DEV_TWO_PLAYER_LINEUP,
}

LOCAL_BOT_CONFIG = {
    "actTimeout": 30.0,
    "agentTimeout": 30.0,
    "runTimeout": 3600.0,
}


def current_lineup():
    return LINEUPS.get(current_match_mode, FOUR_PLAYER_LINEUP)


def use_raw_actions():
    return current_match_mode == "dev2p"


def load_bot_agent():
    global bot_agent
    return load_named_agent(bot_name)


def load_named_agent(name):
    if name in orbit_wars.agents:
        return orbit_wars.agents[name]
    if name == "medium":
        return load_medium_agent()
    if name in PUBLIC_BOTS:
        return load_public_agent(name)
    if name == "main.py":
        return load_local_agent(reload=True)
    if name == "producer_flow_bot.py":
        return load_python_file_agent("producer_flow_bot.py", reload=True)
    return orbit_wars.agents["starter"]


def load_public_agent(name):
    if name in public_agents:
        return public_agents[name]

    bot_path = PUBLIC_BOTS[name]
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), bot_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    relax_local_bot_limits(module)
    public_modules[name] = module
    public_agents[name] = getattr(module, "agent")
    return public_agents[name]


def relax_local_bot_limits(module):
    for attr in ("SOFT_ACT_DEADLINE", "HEAVY_PHASE_MIN_TIME", "OPTIONAL_PHASE_MIN_TIME"):
        if hasattr(module, attr):
            setattr(module, attr, 30.0)


def reset_agent_state():
    for module in public_modules.values():
        if hasattr(module, "_agent_step"):
            setattr(module, "_agent_step", 0)
    if medium_agent is not None and hasattr(medium_agent, "_agent_step"):
        setattr(medium_agent, "_agent_step", 0)


def run_bot_agent(agent, obs):
    signature = inspect.signature(agent)
    accepts_config = (
        any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in signature.parameters.values())
        or len(signature.parameters) >= 2
    )
    if accepts_config:
        return agent(obs, LOCAL_BOT_CONFIG)
    return agent(obs)


def load_medium_agent():
    global medium_agent
    if medium_agent is not None:
        return medium_agent

    medium_path = ROOT / "medium_bot.py"
    spec = importlib.util.spec_from_file_location("orbit_wars_medium_agent", medium_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    medium_agent = getattr(module, "agent")
    return medium_agent


def load_local_agent(reload=False):
    global bot_agent
    if bot_agent is not None and not reload:
        return bot_agent

    main_path = ROOT / "main.py"
    module_name = f"orbit_wars_main_agent_{random.randrange(2**31)}" if reload else "orbit_wars_main_agent"
    spec = importlib.util.spec_from_file_location(module_name, main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    bot_agent = getattr(module, "agent")
    return bot_agent


def load_python_file_agent(filename, reload=False):
    module_path = ROOT / filename
    module_name = (
        f"orbit_wars_{module_path.stem}_{random.randrange(2**31)}"
        if reload
        else f"orbit_wars_{module_path.stem}"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, "agent")


def load_local_module_for_analysis():
    main_path = ROOT / "main.py"
    module_name = f"orbit_wars_main_analysis_{random.randrange(2**31)}"
    spec = importlib.util.spec_from_file_location(module_name, main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def to_plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    if isinstance(value, tuple):
        return [to_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if hasattr(value, "items"):
        return {str(k): to_plain(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {k: to_plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def current_payload():
    obs = to_plain(env.state[0].observation)
    lineup = current_lineup()
    players = []
    for idx, state in enumerate(env.state):
        player_obs = to_plain(state.observation)
        players.append(
            {
                "player": player_obs.get("player"),
                "name": lineup[idx][1] if idx < len(lineup) else f"P{idx}",
                "bot": lineup[idx][0] if idx < len(lineup) else "",
                "reward": state.reward,
                "status": state.status,
                "score": score_player(player_obs, player_obs.get("player")),
            }
        )
    payload = {
        "observation": obs,
        "players": players,
        "done": bool(env.done),
        "info": to_plain(getattr(env, "info", {})),
        "seed": to_plain(getattr(env, "info", {})).get("seed"),
        "bot": bot_name,
        "matchMode": current_match_mode,
        "lineup": [{"bot": bot, "name": name} for bot, name in lineup],
        "actionHistory": to_plain(action_history),
    }
    if current_match_mode == "dev2p":
        payload["quadrantRegions"] = build_quadrant_debug_payload()
    return payload


def build_quadrant_debug_payload():
    try:
        module = load_local_module_for_analysis()
        context = module.build_context(env.state[0].observation)
        report = context["start_region_value_report"]["quadrants"]
        return {
            "priorityGroups": report.get("priority_groups"),
            "rotatingGroups": report.get("rotating_groups"),
            "priorityTarget": report["my_region"].get("priority_target"),
        }
    except Exception as exc:
        app.logger.exception("Quadrant debug payload failed: %s", exc)
        return None


def score_player(obs, player):
    if player is None:
        return 0
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    return sum(int(p[5]) for p in planets if int(p[1]) == player) + sum(
        int(f[6]) for f in fleets if int(f[1]) == player
    )


def sanitize_moves(moves):
    clean = []
    if not isinstance(moves, list):
        return clean
    for move in moves:
        if not isinstance(move, (list, tuple)) or len(move) != 3:
            continue
        try:
            from_id = move[0]
            angle = float(move[1])
            ships = int(move[2])
        except (TypeError, ValueError):
            continue
        if (
            not isinstance(from_id, int)
            or isinstance(from_id, bool)
            or not math.isfinite(angle)
            or ships <= 0
        ):
            continue
        clean.append([from_id, angle, ships])
    return clean


def raw_env_actions(moves):
    return moves if isinstance(moves, list) else []


def describe_actions(step, player_index, bot_id, bot_name, obs, moves):
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    planet_by_id = {int(p[0]): p for p in planets}
    entries = []
    for move in moves:
        source_id, angle, ships = move
        source = planet_by_id.get(int(source_id))
        target = infer_target(source, angle, planets)
        entry = {
            "step": step,
            "player": player_index,
            "bot": bot_id,
            "name": bot_name,
            "source": int(source_id),
            "ships": int(ships),
            "angle": float(angle),
            "angleDeg": round(math.degrees(float(angle)), 1),
            "target": int(target[0]) if target is not None else None,
        }
        entries.append(entry)
    if not entries:
        entries.append(
            {
                "step": step,
                "player": player_index,
                "bot": bot_id,
                "name": bot_name,
                "source": None,
                "ships": 0,
                "angle": None,
                "angleDeg": None,
                "target": None,
            }
        )
    return entries


def infer_target(source, angle, planets):
    if source is None:
        return None
    sx, sy = float(source[2]), float(source[3])
    best = None
    best_score = float("inf")
    for planet in planets:
        if int(planet[0]) == int(source[0]):
            continue
        dx = float(planet[2]) - sx
        dy = float(planet[3]) - sy
        distance = math.hypot(dx, dy)
        if distance <= 0:
            continue
        target_angle = math.atan2(dy, dx)
        delta = abs((target_angle - angle + math.pi) % (2 * math.pi) - math.pi)
        if delta > 0.5:
            continue
        score = delta * 100 + distance
        if score < best_score:
            best = planet
            best_score = score
    return best


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.post("/api/new")
def new_game():
    global env, action_history, current_match_mode, bot_agent
    data = request.get_json(silent=True) or {}
    requested_mode = str(data.get("matchMode") or data.get("mode") or current_match_mode)
    current_match_mode = requested_mode if requested_mode in LINEUPS else "4p"
    lineup = current_lineup()
    if current_match_mode == "dev2p":
        bot_agent = None
    for bot_id, _ in lineup:
        load_named_agent(bot_id)
    reset_agent_state()
    raw_seed = data.get("seed")
    seed = None
    if raw_seed not in (None, ""):
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            seed = None
    if seed is None:
        seed = random.randrange(2**31)

    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.reset(len(lineup))
    action_history = []
    return jsonify(current_payload())


@app.post("/api/step")
def step_game():
    global env
    if env is None or env.done:
        return jsonify({"error": "No active game. Start a new game first."}), 400

    actions = []
    turn_entries = []
    current_step = to_plain(env.state[0].observation).get("step", 0)
    lineup = current_lineup()
    for idx, state in enumerate(env.state):
        bot_id = lineup[idx][0]
        display_name = lineup[idx][1]
        try:
            moves = run_bot_agent(load_named_agent(bot_id), state.observation)
        except Exception as exc:
            moves = []
            app.logger.exception("Bot %s action failed: %s", bot_id, exc)
        clean_moves = sanitize_moves(moves)
        actions.append(raw_env_actions(moves) if use_raw_actions() else clean_moves)
        turn_entries.extend(
            describe_actions(
                current_step,
                idx,
                bot_id,
                display_name,
                to_plain(state.observation),
                clean_moves,
            )
        )
    try:
        env.step(actions)
    except Exception as exc:
        app.logger.exception("Environment step failed")
        return (
            jsonify(
                {
                    "error": "Environment step failed",
                    "detail": str(exc),
                    "matchMode": current_match_mode,
                }
            ),
            400,
        )
    action_history.extend(turn_entries)
    return jsonify(current_payload())


@app.get("/api/state")
def state():
    if env is None:
        return jsonify({"error": "No active game. Start a new game first."}), 400
    return jsonify(current_payload())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
