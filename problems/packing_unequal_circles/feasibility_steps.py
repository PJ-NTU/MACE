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

    for i, c in enumerate(coords):
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            return False, f"coords[{i}] must be a (x, y) pair"

    # Identify packed circles (not marked as (-1, -1))
    packed_indices = []
    for i in range(n):
        x, y = coords[i]
        if x != -1 or y != -1:
            packed_indices.append(i)

    # (5) cross-element / global constraints

    # Prefix property
    if packed_indices:
        K = max(packed_indices)
        packed_set = set(packed_indices)
        for i in range(K):
            if i not in packed_set:
                return False, f"prefix property violated: circle {i} is not packed while circle {K} is packed"

    # Container containment for each packed circle
    for i in packed_indices:
        x, y = coords[i]
        r = radii[i]
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        clearance = R - (dist + r)
        if clearance < -tol:
            return False, f"circle {i} violates container constraint by {-clearance}"

    # Non-overlap for each pair of packed circles
    for idx, i in enumerate(packed_indices):
        for j in packed_indices[idx + 1:]:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            center_distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            required_distance = radii[i] + radii[j]
            if center_distance - required_distance < -tol:
                return False, f"circles {i} and {j} overlap by {required_distance - center_distance}"

    return True, None
'''
