# Orbit Wars Bot Core Rules

This document captures the core rules and bot-development assumptions we are using for the local Orbit Wars setup. It is meant to be stable reference material before we later rebuild the bot into a higher-performance search-style engine.

## Game Basics

- Board size is `100 x 100`.
- Board center is `(50, 50)`.
- The sun is centered at `(50, 50)` with radius `10`.
- Game length is `500` steps.
- The game supports `2` or `4` players.
- A player wins by having the highest total ships at the end, counting ships on owned planets plus ships in owned fleets.
- A player can also win earlier if only that player remains with planets/fleets.

## Planets

Each planet is represented as:

```python
[id, owner, x, y, radius, ships, production]
```

Fields:

- `id`: unique planet ID.
- `owner`: `-1` for neutral, otherwise player ID.
- `x`, `y`: current position.
- `radius`: collision/capture radius.
- `ships`: ships currently stationed on the planet.
- `production`: ships generated each turn if the planet is owned.

Owned planets produce ships every turn:

```python
planet.ships += planet.production
```

Neutral planets do not produce.

## Moving Planets

Some planets rotate around the sun. Static vs rotating is determined by:

```python
orbit_radius + planet.radius >= 50
```

If true, the planet is static. Otherwise, it rotates around the center.

Future rotating position can be predicted using:

```python
future_angle = current_angle + angular_velocity * turns
x = 50 + orbit_radius * cos(future_angle)
y = 50 + orbit_radius * sin(future_angle)
```

Accurate shots against rotating planets require intercept calculation, not just aiming at current position.

## Comets

Comets are temporary planets.

Important comet rules:

- Comets appear in the normal `planets` list.
- Comets can be captured.
- Owned comets produce ships.
- Fleets can launch from owned comets.
- Comets move along predefined paths.
- When a comet leaves the board, it is removed.
- Ships stationed on a removed comet are lost.

Comet IDs are available from:

```python
obs["comet_planet_ids"]
```

Active comet path data is available from:

```python
obs["comets"]
```

## Fleets

Each fleet is represented as:

```python
[id, owner, x, y, angle, from_planet_id, ships]
```

Fields:

- `id`: unique fleet ID.
- `owner`: player ID.
- `x`, `y`: current fleet position.
- `angle`: movement direction in radians.
- `from_planet_id`: source planet ID.
- `ships`: ships in the fleet.

Fleet speed depends on fleet size:

```python
speed = 1.0 + (6.0 - 1.0) * (log(ships) / log(1000)) ** 1.5
```

Speed is capped by the formula at approximately `6.0`.

## Actions

Each bot returns a list of moves:

```python
[[from_planet_id, angle_in_radians, num_ships], ...]
```

A move launches `num_ships` from an owned planet in the given angle.

The environment only launches if:

- action is a list
- move has exactly 3 items
- source planet exists
- source planet is owned by the player
- `ships` can be converted to int
- `ships > 0`
- source planet has enough ships

Invalid moves are effectively ignored.

## Combat

When a fleet reaches a planet:

- If same owner: surviving ships are added to the planet.
- If different owner: fleet ships subtract from garrison.
- If attacking ships exceed garrison, ownership flips.
- New garrison becomes the surplus attacking ships.

Capture cost baseline:

```python
target.ships + 1
```

This does not include future production, incoming fleets, or multi-party combat. Stronger logic should account for those.

## Destruction

Fleets are destroyed if they:

- hit the sun
- leave the board

The board does not wrap around. A fleet leaving one side does not return from the opposite side.

## Turn Order Notes

Important practical turn-order assumptions:

- Actions are collected from all players.
- Valid launches create fleets.
- Owned planets, including owned comets, produce ships.
- Planets/comets move.
- Fleet collisions and combat are resolved by the environment.
- Comets that leave the board are removed, including ships on them.

For exact details, trust the official Kaggle `orbit_wars` environment in local execution.

## Bot Observation Data

The bot receives `obs`.

Common fields:

- `player`: current player ID.
- `step`: current game step.
- `planets`: all planets, including comets.
- `fleets`: all active fleets.
- `angular_velocity`: rotation speed for rotating planets.
- `initial_planets`: starting planet positions.
- `comets`: active comet path data.
- `comet_planet_ids`: IDs of planets that are comets.
- `remainingOverageTime`: extra time budget for Kaggle submissions.

## Current Development Setup

Local browser modes:

- `4P`: Public 1224 / Public 1000 / Public 1060 / Starter
- `2P`: Public 1224 vs Public 1060
- `Dev 2P`: `main.py` vs Public 1060

`Dev 2P` is the default screen. It reloads `main.py` on each step so bot edits can be tested without restarting the server.

## Current Bot Strategy State

Current `main.py` is a scaffold, not a final strong bot.

Implemented helper concepts:

- planet/fleet parsing
- my/enemy/neutral grouping
- score by player
- fleet speed
- travel-time estimate
- capture cost
- static vs rotating planet detection
- future rotating planet position
- comet future position/lifetime
- sun path safety
- exact discrete intercept search
- region grouping
- first-10-step region value report

Current active decision rule:

- Use only the quadrant-based region report.
- Select the quadrant priority target.
- Wait until enough ships are available.
- Use exact intercept aiming.
- Launch only if exact hit timing is found.

## Region Analysis

The map is analyzed in three ways:

```python
quadrants:
  top_left
  top_right
  bottom_left
  bottom_right

vertical_halves:
  left
  right

horizontal_halves:
  top
  bottom
```

For now, only `quadrants` are used for actual decisions.

For each region, the bot computes:

- total capture cost
- total production revenue
- ranked target planets
- top target for each of the first 10 player steps
- priority target

Current waiting-step formula:

```python
waiting_steps = ceil((target_cost - current_bot_planet_fleet + 1) / bot_planet_production)
```

If already affordable, waiting steps is `0`.

Current priority formula:

```python
priority_value = (20 - n_step - effectiveness) * target_production
```

Where:

- `n_step` is `1` through `10`
- `effectiveness` is currently waiting steps
- larger `priority_value` wins

## Exact Intercept Rule

The bot should not accept early or late hits as valid planning.

A planned shot is valid only when:

```python
simulated_hit_step == planned_flight_steps
```

This avoids wasting a step by chasing behind a rotating planet, and avoids false positives where the fleet collides earlier than the planned target time.

Exact intercept search:

1. Choose a candidate flight duration.
2. Predict target position at that arrival duration.
3. Compute angle from source to that future target position.
4. Simulate fleet movement step by step.
5. Simulate target movement step by step.
6. Reject if fleet hits sun or goes out of bounds.
7. Reject if first collision is earlier or later than planned.
8. Accept only exact timing.

## Future High-Performance Direction

If we later rebuild this like a search engine, useful architecture pieces:

- immutable compact game state
- fast simulator matching Kaggle rules
- legal/valid action generator
- target-action candidate generator
- exact intercept cache
- region/economy evaluator
- threat evaluator
- rollout/search loop
- transposition cache keyed by compact state
- time manager
- deterministic seed-free behavior

Possible scoring terms:

- total ships
- production controlled
- projected production after capture
- capture timing
- sun/path risk
- incoming enemy threats
- overkill waste
- region control
- comet lifetime risk
- enemy denial value
- endgame immediate score value

The current scaffold should be treated as the readable prototype layer before any optimized rewrite.
