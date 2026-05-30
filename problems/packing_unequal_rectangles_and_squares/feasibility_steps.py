"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    tol = 1e-5

    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    if "placements" not in solution:
        return False, "solution missing 'placements' key"

    placements = solution["placements"]

    # (3) correct type and length
    if not isinstance(placements, list):
        return False, f"'placements' must be list, got {type(placements).__name__}"
    if len(placements) != n:
        return False, f"placements length {len(placements)} != n={n}"

    packed_rectangles = []  # (xmin, xmax, ymin, ymax) for overlap checking
    packed_indices = []

    for idx, placement in enumerate(placements):
        # (3) each placement has correct type and length
        if not isinstance(placement, (list, tuple)) or len(placement) != 3:
            return False, f"placement {idx} must be a tuple/list of length 3 (x, y, theta)"

        x, y, theta = placement

        # unpacked item — skip further checks
        if x == -1 and y == -1:
            continue

        # (4) rotation angle constraints
        if rotation:
            if not (math.isclose(theta, 0, abs_tol=1e-3) or math.isclose(theta, 90, abs_tol=1e-3)):
                return False, f"item {idx}: theta must be 0 or 90 when rotation is allowed, got {theta}"
        else:
            if not math.isclose(theta, 0, abs_tol=1e-3):
                return False, f"item {idx}: theta must be 0 when rotation is not allowed, got {theta}"

        # (4) square dimension consistency
        L, W = items[idx]
        if shape.lower() == "square" and not math.isclose(L, W, abs_tol=1e-3):
            return False, f"item {idx}: square packing requires equal dimensions, got ({L}, {W})"

        # effective dimensions after optional rotation
        if rotation and math.isclose(theta, 90, abs_tol=1e-3):
            eff_L, eff_W = W, L
        else:
            eff_L, eff_W = L, W

        half_L = eff_L / 2.0
        half_W = eff_W / 2.0

        # (5) all four corners must lie inside the circular container
        corners = [
            (x - half_L, y - half_W),
            (x - half_L, y + half_W),
            (x + half_L, y - half_W),
            (x + half_L, y + half_W),
        ]
        for corner in corners:
            dist = math.hypot(corner[0] - cx, corner[1] - cy)
            if dist > R + tol:
                return False, f"item {idx}: corner {corner} lies outside the container (dist={dist:.6f} > R={R})"

        # (5) no overlap with previously packed items
        xmin, xmax = x - half_L, x + half_L
        ymin, ymax = y - half_W, y + half_W
        for jdx, (oxmin, oxmax, oymin, oymax) in zip(packed_indices, packed_rectangles):
            if not (xmax <= oxmin + tol or xmin >= oxmax - tol or
                    ymax <= oymin + tol or ymin >= oymax - tol):
                return False, f"item {idx} overlaps with packed item {jdx}"

        packed_rectangles.append((xmin, xmax, ymin, ymax))
        packed_indices.append(idx)

    return True, None
'''
