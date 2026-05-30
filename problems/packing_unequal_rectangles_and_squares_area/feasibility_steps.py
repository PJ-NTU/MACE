"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    tol = 1e-5
    angle_tol = 1e-3

    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    placements = solution.get("placements")
    if placements is None:
        return False, "solution missing 'placements' key"

    # (3) correct type and length
    if not isinstance(placements, (list, tuple)):
        return False, f"'placements' must be list, got {type(placements).__name__}"
    if len(placements) != n:
        return False, f"'placements' length {len(placements)} != n={n}"

    # (4) per-element value constraints
    placed_items = []
    for i in range(n):
        placement = placements[i]
        if not isinstance(placement, (list, tuple)) or len(placement) != 3:
            return False, f"placement[{i}] must be a tuple/list of 3 numbers"
        try:
            x, y, theta = float(placement[0]), float(placement[1]), float(placement[2])
        except (TypeError, ValueError):
            return False, f"placement[{i}] entries must be numeric"

        L, W = items[i]

        # unpacked item
        if abs(x + 1) < tol and abs(y + 1) < tol:
            if abs(theta) > angle_tol:
                return False, f"unpacked item {i} must have theta=0, got {theta}"
            continue

        # packed item: check rotation
        if not rotation:
            if abs(theta) > angle_tol:
                return False, f"rotation not allowed but item {i} has theta={theta}"
        else:
            if not (abs(theta) < angle_tol or abs(theta - 90) < angle_tol):
                return False, f"item {i} has invalid theta={theta}; allowed: 0 or 90"

        # compute vertices
        import math
        t = math.radians(theta)
        cos_t, sin_t = math.cos(t), math.sin(t)
        local = [(L/2, W/2), (L/2, -W/2), (-L/2, W/2), (-L/2, -W/2)]
        vertices = [(x + dx*cos_t - dy*sin_t, y + dx*sin_t + dy*cos_t) for dx, dy in local]

        # check all vertices inside container
        for vx, vy in vertices:
            if (vx - cx)**2 + (vy - cy)**2 > R**2 + tol:
                return False, f"item {i} vertex ({vx:.4f},{vy:.4f}) outside container"

        # compute aabb
        if abs(theta) < angle_tol:
            half_L, half_W = L/2, W/2
        else:
            half_L, half_W = W/2, L/2
        aabb = (x - half_L, x + half_L, y - half_W, y + half_W)
        placed_items.append({'index': i, 'aabb': aabb})

    # (5) cross-element: no pairwise overlap
    for a in range(len(placed_items)):
        xmin_i, xmax_i, ymin_i, ymax_i = placed_items[a]['aabb']
        for b in range(a + 1, len(placed_items)):
            xmin_j, xmax_j, ymin_j, ymax_j = placed_items[b]['aabb']
            overlap_x = max(0.0, min(xmax_i, xmax_j) - max(xmin_i, xmin_j))
            overlap_y = max(0.0, min(ymax_i, ymax_j) - max(ymin_i, ymin_j))
            if overlap_x * overlap_y > tol:
                return False, f"items {placed_items[a]['index']} and {placed_items[b]['index']} overlap"

    return True, None
'''
