"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    import math

    tol = 1e-5

    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    if "coords" not in solution:
        return False, "solution missing 'coords' key"

    coords = solution["coords"]

    # (3) correct type
    if not isinstance(coords, (list, tuple)):
        return False, f"'coords' must be list, got {type(coords).__name__}"

    # (4) per-element value constraints
    if len(coords) != n:
        return False, f"coords length {len(coords)} != n={n}"

    for i, entry in enumerate(coords):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return False, f"coords[{i}] must be a (x, y) pair"

    # Identify packed circles
    packed_indices = []
    for i in range(n):
        x, y = coords[i]
        if not (abs(x + 1) <= tol and abs(y + 1) <= tol):
            packed_indices.append(i)

    # (5) cross-element / global constraints

    # Container containment for each packed circle
    for i in packed_indices:
        x, y = coords[i]
        r = radii[i]
        dist = math.hypot(x - cx, y - cy)
        clearance = R - (dist + r)
        if clearance < -tol:
            return False, f"Circle {i} violates container constraint by {-clearance}"

    # Non-overlap for every pair of packed circles
    for idx, i in enumerate(packed_indices):
        for j in packed_indices[idx + 1:]:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            center_distance = math.hypot(x1 - x2, y1 - y2)
            required_distance = radii[i] + radii[j]
            clearance = center_distance - required_distance
            if clearance < -tol:
                return False, f"Circles {i} and {j} overlap by {-clearance}"

    return True, None
'''
