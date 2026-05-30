"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    import math

    TOL = 1e-6

    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    if "steiner_points" not in solution:
        return False, "solution missing 'steiner_points' key"

    # (3) correct type for steiner_points
    steiner_points = solution["steiner_points"]
    if not isinstance(steiner_points, (list, tuple)):
        return False, f"'steiner_points' must be list or tuple, got {type(steiner_points).__name__}"

    # (4) per-element value constraints
    for i, sp in enumerate(steiner_points):
        if not isinstance(sp, (list, tuple)) or len(sp) != 2:
            return False, f"steiner_points[{i}] must be a (x, y) tuple/list of length 2"
        x, y = sp
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False, f"steiner_points[{i}] coordinates must be numeric"

    # (5) cross-element / global constraint: MST over union must not exceed MST over originals
    def euclidean_distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def compute_mst_length(pts):
        n = len(pts)
        if n == 0:
            return 0.0
        in_mst = [False] * n
        min_dist = [float('inf')] * n
        min_dist[0] = 0.0
        total = 0.0
        for _ in range(n):
            u = -1
            best = float('inf')
            for j in range(n):
                if not in_mst[j] and min_dist[j] < best:
                    best = min_dist[j]
                    u = j
            if u == -1:
                break
            in_mst[u] = True
            total += best
            for v in range(n):
                if not in_mst[v]:
                    d = euclidean_distance(pts[u], pts[v])
                    if d < min_dist[v]:
                        min_dist[v] = d
        return total

    mst_original = compute_mst_length(points)
    union_pts = list(points) + [tuple(sp) for sp in steiner_points]
    candidate_value = compute_mst_length(union_pts)

    if candidate_value > mst_original + TOL:
        return False, (
            f"candidate MST length ({candidate_value}) exceeds original MST length "
            f"({mst_original}); Steiner points must not increase the MST"
        )

    return True, None
'''
