"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    if "total_value" not in solution:
        return False, "solution missing 'total_value' key"
    if "placements" not in solution:
        return False, "solution missing 'placements' key"

    # (3) correct types for top-level keys
    total_value = solution["total_value"]
    placements = solution["placements"]
    if not isinstance(total_value, int):
        return False, f"'total_value' must be int, got {type(total_value).__name__}"
    if not isinstance(placements, (list, tuple)):
        return False, f"'placements' must be list, got {type(placements).__name__}"

    # (4) per-element constraints
    rects = []
    type_counts = [0] * m
    computed_value = 0

    for idx, placement in enumerate(placements):
        if not isinstance(placement, (list, tuple)) or len(placement) != 6:
            return False, f"placement {idx} must be a 6-element list/tuple, got {placement}"
        try:
            type_idx = int(placement[0])
            x        = int(placement[1])
            y        = int(placement[2])
            placed_len = int(placement[3])
            placed_wid = int(placement[4])
            orient   = int(placement[5])
        except Exception:
            return False, f"placement {idx} contains non-integer values: {placement}"

        if type_idx < 1 or type_idx > m:
            return False, f"placement {idx} has invalid piece type index {type_idx} (must be 1..{m})"
        if orient != 0:
            return False, f"placement {idx} has orientation flag {orient}; must be 0 (no rotation)"

        piece = piece_types[type_idx - 1]
        if placed_len != piece['length'] or placed_wid != piece['width']:
            return False, (f"placement {idx} dimensions ({placed_len}, {placed_wid}) do not match "
                           f"expected ({piece['length']}, {piece['width']}) for piece type {type_idx}")

        if x < 0 or y < 0 or (x + placed_len) > stock_length or (y + placed_wid) > stock_width:
            return False, (f"placement {idx} rectangle ({x},{y},{x+placed_len},{y+placed_wid}) "
                           f"is out of stock bounds (0,0)-({stock_length},{stock_width})")

        type_counts[type_idx - 1] += 1
        computed_value += piece['value']
        rects.append((x, y, x + placed_len, y + placed_wid))

    # (5) cross-element / global constraints

    # No overlaps
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            r1, r2 = rects[i], rects[j]
            if min(r1[2], r2[2]) - max(r1[0], r2[0]) > 0 and min(r1[3], r2[3]) - max(r1[1], r2[1]) > 0:
                return False, f"placements {i} and {j} overlap"

    # Max count per piece type
    for i in range(m):
        if type_counts[i] > piece_types[i]['max']:
            return False, (f"piece type {i+1} placed {type_counts[i]} times, "
                           f"exceeds max {piece_types[i]['max']}")

    # Guillotine condition
    def is_guillotine(rects, bx, by, ex, ey):
        if not rects:
            return True
        if len(rects) == 1:
            r = rects[0]
            if r[0] == bx and r[1] == by and r[2] == ex and r[3] == ey:
                return True
        for x in range(bx + 1, ex):
            if all(r[2] <= x or r[0] >= x for r in rects):
                if is_guillotine([r for r in rects if r[2] <= x], bx, by, x, ey) and \
                   is_guillotine([r for r in rects if r[0] >= x], x, by, ex, ey):
                    return True
        for y in range(by + 1, ey):
            if all(r[3] <= y or r[1] >= y for r in rects):
                if is_guillotine([r for r in rects if r[3] <= y], bx, by, ex, y) and \
                   is_guillotine([r for r in rects if r[1] >= y], bx, y, ex, ey):
                    return True
        return False

    if not is_guillotine(rects, 0, 0, stock_length, stock_width):
        return False, "placement layout is not guillotine separable"

    # Reported total_value must match computed sum
    if total_value != computed_value:
        return False, f"reported total_value {total_value} != computed value {computed_value}"

    return True, None
'''
