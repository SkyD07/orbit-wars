# Bot Performance Tradeoffs

This note records why we are considering a faster internal data model, what problem it is meant to solve, and what risks it introduces. Keep this rationale visible so future debugging does not erase the original purpose.

## Current Position

The active development bot is `producer_flow_bot.py`.

It currently favors readable structures:

```python
Planet(id, owner, x, y, radius, ships, production)
Fleet(id, owner, x, y, angle, from_planet_id, ships)
Candidate(source_id, target_id, ships, angle, eta, score, mission)
```

These namedtuples are created from raw Kaggle arrays. The values are raw observation values, but the container format is our helper format.

This is not expected to be the main bottleneck yet. Orbit Wars usually has a small planet count and a manageable fleet count. The heavier work is likely:

```text
future planet position calls
incoming fleet hit prediction
candidate ETA estimation
path validation
first planet collision checks
candidate scoring and sorting
```

## Original Purpose

The goal is not to make the code look optimized. The goal is to make the bot stronger under Kaggle's action time limit by spending compute on useful tactical decisions.

The intended bot style is:

```text
fast flow-diff planning
many cheap candidate tests
bounded geometric validation
greedy action allocation under a policy budget
```

Any performance rewrite must preserve this strategic shape.

## Fast Data Model Idea

A faster version would use raw arrays, integer indexes, and compact caches.

Instead of:

```python
p.x
p.production
p.ships
```

it would use index constants:

```python
P_ID = 0
P_OWNER = 1
P_X = 2
P_Y = 3
P_RADIUS = 4
P_SHIPS = 5
P_PROD = 6
```

Or per-turn packed arrays:

```python
p_id[i]
p_owner[i]
p_x[i]
p_y[i]
p_radius[i]
p_ships[i]
p_prod[i]
```

Algorithms would operate on planet indexes:

```python
for si in my_idx:
    for ti in target_idx:
        ...
```

## Benefits

Potential benefits:

```text
less object allocation
faster hot loops
faster tuple sorting for candidates
better cache locality
easier distance/future-position matrix caches
lower overhead in candidate generation
lower overhead in path validation
```

This matters if we later run thousands of candidates and hundreds of validations per turn.

## Risks

The main risk is correctness.

Raw indexes are easy to mix up:

```python
p[5]  # ships
p[6]  # production
```

Other common bug classes:

```text
planet id vs planet index confusion
source index vs source id confusion
target index vs target id confusion
using current position where initial position is required
forgetting comet path special cases
forgetting source reserve or action policy caps
breaking collision correctness while optimizing
```

These bugs can be subtle because the bot may still return legal actions while making strategically bad moves.

## Readability Cost

Readable code:

```python
target.production * remaining
source.ships
target.owner
```

Fast code:

```python
p_prod[ti] * remaining
p_ships[si]
p_owner[ti]
```

The fast version is harder for humans to review. It also makes strategic discussion harder because the code no longer names the domain concepts directly.

## Recommended Decision Rule

Do not optimize away readability unless profiling shows a real need.

Prefer this order:

```text
1. cache future positions
2. cache distances and center distances
3. reduce validation work
4. use lighter candidate tuples
5. spatially prune path validation
6. only then consider full raw-array/index rewrite
```

If the bot is comfortably under the time limit, keep namedtuples.

If performance becomes tight, optimize only the hot path first.

## Non-Negotiable Correctness Rules

Any faster implementation must preserve:

```text
ships_sent is not ships_lost
source reserve protection
policy action budget
ship buckets are alternatives, not independent actions
first planet collision must be the intended target
sun collision rejection
out-of-bounds rejection
comet lifetime handling
event-based target forecast
```

Performance work is not successful if it makes the bot faster but less faithful to these rules.

