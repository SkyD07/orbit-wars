# Fast Data Implementation Plan

This document describes how to move toward a faster data model without losing the original purpose of the bot. It is intentionally phased. Do not jump directly to a full rewrite unless profiling justifies it.

## Goal

Improve performance only where it helps the active bot:

```text
ProducerFlowBot
fast candidate generation
cheap forecasting
bounded validation
policy-limited action selection
```

The goal is not to replace the strategy. The goal is to make the existing strategy cheaper to run.

## Baseline To Preserve

Before any optimization, record behavior on a small seed set:

```text
1
42
1227252659
20260612
987654321
```

For each seed, observe:

```text
first move step
number of moves in first 30 steps
obvious wrong-planet collisions
opening expansion quality
source planets drained or preserved
whether bot stalls
```

Any performance change should be compared against this baseline.

## Phase 1: Add Index Constants

Add constants for raw Kaggle arrays:

```python
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
```

Purpose:

```text
remove magic indexes before raw-array work starts
make future optimized code less error-prone
```

Risk:

```text
low
```

Validation:

```text
compile
same smoke seeds
no behavior change expected
```

## Phase 2: Add Cheap Caches Without Rewriting Structures

Keep `Planet` and `Fleet` namedtuples, but add caches:

```text
distance_to_center_by_planet_id
static_flag_by_planet_id
future_position_cache[(planet_id, turns)]
fleet_speed_cache[ships]
```

Purpose:

```text
reduce repeated geometry work
keep code readable
```

Risk:

```text
medium-low
```

Important cache keys:

```text
planet_id
turn offset
current step if needed
angular_velocity
comet path index if comet
```

Validation:

```text
future positions match old function
path validation behavior unchanged except faster
```

## Phase 3: Candidate Tuple Optimization

Replace `Candidate` namedtuple with tuple-based candidates only if candidate count becomes expensive.

Possible tuple:

```python
(score, source_id, target_id, ships, angle, eta, mission)
```

Purpose:

```text
faster sorting
less allocation overhead
```

Risk:

```text
medium
```

Main risk:

```text
field order bugs
```

Mitigation:

```text
define constants for tuple indexes
keep conversion helper for debug printing
```

Validation:

```text
same selected move count and similar target choices on baseline seeds
```

## Phase 4: Add Index Arrays Alongside Namedtuples

Build a fast per-turn data layer in `build_context`.

Example:

```python
ctx["p_raw"] = raw_planets
ctx["p_id"] = [...]
ctx["p_owner"] = [...]
ctx["p_x"] = [...]
ctx["p_y"] = [...]
ctx["p_radius"] = [...]
ctx["p_ships"] = [...]
ctx["p_prod"] = [...]
ctx["id_to_i"] = {planet_id: i}
ctx["my_idx"] = [...]
ctx["enemy_idx"] = [...]
ctx["neutral_idx"] = [...]
```

Keep the old namedtuple paths during this phase.

Purpose:

```text
prepare hot loops for index-based access
allow gradual migration
```

Risk:

```text
medium
```

Main risk:

```text
planet id and planet index confusion
```

Rule:

```text
suffix index variables with _i or use si/ti/pi
suffix ids with _id
```

Validation:

```text
id_to_i[p_id[i]] == i
my_idx matches my_planets ids
enemy_idx matches enemy_planets ids
neutral_idx matches neutral_planets ids
```

## Phase 5: Migrate Hot Candidate Loops

Move only these hot paths first:

```text
candidate_targets_for_source
ship_buckets
score_move
select_moves_greedy
```

Keep path validation readable until the candidate path is stable.

Purpose:

```text
speed up high-volume candidate work
avoid touching collision correctness too early
```

Risk:

```text
medium
```

Validation:

```text
candidate count similar
top candidate targets similar
selected moves legal
baseline seed behavior acceptable
```

## Phase 6: Optimize Path Validation

Path validation is likely the biggest performance target.

Possible improvements:

```text
future position cache by planet index and turn
skip source planet immediately after launch
spatial prune planets far from path segment
only check planets whose bounding box intersects segment bounds
reuse sin/cos/speed per candidate
```

Purpose:

```text
reduce first-collision validation cost
```

Risk:

```text
high
```

Main risk:

```text
approving paths that hit wrong planet first
rejecting valid paths incorrectly
breaking comet/rotating target behavior
```

Validation:

```text
compare old validator vs new validator on many candidates
log disagreements
manually inspect disagreements in browser
```

Do not remove the old validator until disagreement cases are understood.

## Phase 7: Remove Namedtuple Layer Only If Needed

Only after the hot paths are stable, decide whether to remove namedtuples.

Reasons to remove:

```text
profiling shows object access/allocation still matters
bot is near timeout
raw-array version has enough tests
```

Reasons to keep:

```text
performance is already fine
readability matters for strategy work
debugging optimized code slows iteration
```

## Profiling Checklist

Before deeper optimization, measure:

```text
agent total runtime
candidate generation time
incoming forecast time
path validation time
number of candidates generated
number of candidates scanned
number of candidates validated
number of moves selected
```

If path validation dominates, optimize validation first.

If candidate generation dominates, migrate target/scoring loops.

If parsing is tiny, do not remove namedtuples.

## Debug Checklist For Future Bugs

When optimized code behaves strangely, check:

```text
planet id vs planet index
owner index
ships index
production index
source reserve
policy total commit cap
intent conflict key
future position cache key
comet path index
first collision target
sun collision
out-of-bounds
ETA mismatch
```

## Success Criteria

An optimization is successful only if:

```text
runtime improves or enables more useful candidates
legal action rate stays stable
wrong-path launches do not increase
source planets are not drained unexpectedly
bot still follows policy budget
code remains reviewable enough to tune strategy
```

