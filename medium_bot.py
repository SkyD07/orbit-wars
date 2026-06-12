import math

from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, SUN_RADIUS, Planet


BOARD_CENTER = (CENTER, CENTER)


def _get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _point_segment_distance(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _crosses_sun(source, target):
    return (
        _point_segment_distance(BOARD_CENTER, (source.x, source.y), (target.x, target.y))
        < SUN_RADIUS + 0.8
    )


def _target_score(source, target, player):
    dist = max(1.0, _distance(source, target))
    owner_penalty = 12 if target.owner != -1 and target.owner != player else 0
    capture_cost = target.ships + owner_penalty + 1
    value = target.production * 42 + target.radius * 5
    if target.owner != -1 and target.owner != player:
        value += 22
    if _crosses_sun(source, target):
        value -= 120
    return value / dist - capture_cost * 0.18


def agent(obs):
    player = _get(obs, "player", 0)
    planets = [Planet(*p) for p in _get(obs, "planets", [])]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    moves = []
    reserved_targets = set()
    for source in sorted(my_planets, key=lambda p: p.ships, reverse=True):
        available = int(source.ships) - 3
        if available < 2:
            continue

        ranked = sorted(
            (t for t in targets if t.id not in reserved_targets),
            key=lambda t: _target_score(source, t, player),
            reverse=True,
        )

        for target in ranked:
            if _crosses_sun(source, target):
                continue
            needed = int(target.ships) + 1
            if target.owner != -1:
                needed += max(4, target.production * 3)
            else:
                needed += max(0, target.production - 1)
            if needed <= available:
                angle = math.atan2(target.y - source.y, target.x - source.x)
                moves.append([source.id, angle, needed])
                reserved_targets.add(target.id)
                break

    return moves
