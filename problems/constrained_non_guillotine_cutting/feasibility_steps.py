"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    placements = solution.get("placements")
    if placements is None:
        return False, "solution missing 'placements' key"
    if not isinstance(placements, (list, tuple)):
        return False, f"'placements' must be list, got {type(placements).__name__}"

    counts = [0] * len(pieces)
    rects = []

    for idx, placement in enumerate(placements):
        if not isinstance(placement, (list, tuple)) or len(placement) != 4:
            return False, f"placement at index {idx} must be a 4-tuple"

        piece_type, x, y, r = placement

        if not all(isinstance(val, int) for val in (piece_type, x, y, r)):
            return False, f"all values in placement at index {idx} must be integers"

        if piece_type < 1 or piece_type > len(pieces):
            return False, f"placement at index {idx} has invalid piece_type {piece_type}"

        if r not in (0, 1):
            return False, f"placement at index {idx} has invalid rotation flag {r}"

        piece = pieces[piece_type - 1]
        if r == 0:
            p_len = piece['length']
            p_wid = piece['width']
        else:
            p_len = piece['width']
            p_wid = piece['length']

        if x < 0 or y < 0 or (x + p_len) > stock_length or (y + p_wid) > stock_width:
            return False, f"placement at index {idx} is out of stock boundaries"

        rects.append((x, y, x + p_len, y + p_wid))
        counts[piece_type - 1] += 1

    num_rects = len(rects)
    for i in range(num_rects):
        for j in range(i + 1, num_rects):
            a, b = rects[i], rects[j]
            if not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]):
                return False, f"placements at indices {i} and {j} overlap"

    for i, piece in enumerate(pieces):
        if counts[i] < piece['min'] or counts[i] > piece['max']:
            return False, (f"piece type {i + 1} count {counts[i]} does not meet constraints "
                           f"[min: {piece['min']}, max: {piece['max']}]")

    return True, None
'''
