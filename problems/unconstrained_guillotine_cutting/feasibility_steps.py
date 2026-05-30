"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    if "placements" not in solution:
        return False, "solution missing 'placements' key"

    placements = solution["placements"]

    # (3) correct type for placements
    if not isinstance(placements, list):
        return False, f"'placements' must be list, got {type(placements).__name__}"

    used_piece_ids = set()
    rects = []

    # (4) per-element value constraints
    for idx, placement in enumerate(placements):
        if not isinstance(placement, dict):
            return False, f"placement[{idx}] must be dict, got {type(placement).__name__}"

        for key in ("piece_id", "x", "y", "orientation"):
            if key not in placement:
                return False, f"placement[{idx}] missing key '{key}'"

        try:
            piece_id = int(placement["piece_id"])
            x = int(placement["x"])
            y = int(placement["y"])
            orientation = int(placement["orientation"])
        except Exception as e:
            return False, f"placement[{idx}] has invalid format: {e}"

        if piece_id not in pieces:
            return False, f"piece_id {piece_id} not found in pieces"

        if piece_id in used_piece_ids:
            return False, f"duplicate usage of piece_id {piece_id}"
        used_piece_ids.add(piece_id)

        if orientation not in (0, 1):
            return False, f"invalid orientation {orientation} for piece_id {piece_id}; must be 0 or 1"

        if not allow_rotation and orientation != 0:
            return False, f"rotation not allowed but piece_id {piece_id} has orientation {orientation}"

        if orientation == 0:
            p_width = pieces[piece_id]['l']
            p_height = pieces[piece_id]['w']
        else:
            p_width = pieces[piece_id]['w']
            p_height = pieces[piece_id]['l']

        if x < 0 or y < 0 or (x + p_width) > stock_width or (y + p_height) > stock_height:
            return False, f"piece_id {piece_id} placement is out of stock boundaries"

        rects.append({"x": x, "y": y, "width": p_width, "height": p_height})

    # (5) cross-element / global constraints: no overlaps
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            r1, r2 = rects[i], rects[j]
            x_overlap = max(0, min(r1["x"] + r1["width"], r2["x"] + r2["width"]) - max(r1["x"], r2["x"]))
            y_overlap = max(0, min(r1["y"] + r1["height"], r2["y"] + r2["height"]) - max(r1["y"], r2["y"]))
            if x_overlap * y_overlap > 0:
                return False, f"overlap detected between placement {i} and placement {j}"

    return True, None
'''
