"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    TOL = 1e-6
    container_L, container_W, container_H = container

    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    for key in ("instance", "util", "m", "placements"):
        if key not in solution:
            return False, f"solution missing '{key}' key"

    placements = solution["placements"]
    m = solution["m"]

    # (3) correct types
    if not isinstance(placements, list):
        return False, f"'placements' must be list, got {type(placements).__name__}"
    if not isinstance(m, int):
        return False, f"'m' must be int, got {type(m).__name__}"

    # (4) per-element value constraints
    placed = []
    usage = [0] * len(box_types)

    for idx, placement in enumerate(placements):
        if not isinstance(placement, dict):
            return False, f"placement {idx} must be dict"
        for key in ("box_type", "orientation", "x", "y", "z"):
            if key not in placement:
                return False, f"placement {idx} missing '{key}' key"

        bt_index = placement["box_type"] - 1
        if bt_index < 0 or bt_index >= len(box_types):
            return False, f"Invalid box type index in placement {idx}: {placement['box_type']}"

        orientation = placement["orientation"]
        if orientation not in (1, 2, 3):
            return False, f"Invalid orientation {orientation} in placement {idx}; must be 1, 2, or 3"

        usage[bt_index] += 1
        box = box_types[bt_index]

        try:
            dx, dy, dz, lb, volume = get_box_dimensions(box, orientation)
        except Exception as e:
            return False, f"Orientation error for placement {idx}: {e}"

        x, y, z = placement["x"], placement["y"], placement["z"]
        if (x < -TOL or y < -TOL or z < -TOL or
                x + dx > container_L + TOL or
                y + dy > container_W + TOL or
                z + dz > container_H + TOL):
            return False, f"Box at placement {idx} is out-of-bound"

        placed.append({
            "id": idx,
            "box_type": bt_index,
            "orientation": orientation,
            "x": x, "y": y, "z": z,
            "dx": dx, "dy": dy, "dz": dz,
            "lb": lb,
            "weight": box["weight"],
            "volume": volume,
        })

    # (5a) count constraints
    for i, count in enumerate(usage):
        if count > box_types[i]["count"]:
            return False, (f"Box type {i+1} used {count} times but only "
                           f"{box_types[i]['count']} available")

    # (5b) support constraints
    support_of = {}
    for b in placed:
        if abs(b["z"]) < TOL:
            continue
        candidate = None
        for other in placed:
            if other["id"] == b["id"]:
                continue
            if abs(other["z"] + other["dz"] - b["z"]) > TOL:
                continue
            if b["x"] + TOL < other["x"] or (b["x"] + b["dx"]) - TOL > other["x"] + other["dx"]:
                continue
            if b["y"] + TOL < other["y"] or (b["y"] + b["dy"]) - TOL > other["y"] + other["dy"]:
                continue
            if candidate is not None:
                return False, f"Ambiguous support for box id {b['id']}"
            candidate = other
        if candidate is None:
            return False, f"Missing support for box id {b['id']}"
        support_of[b["id"]] = candidate["id"]

    # (5c) overlap constraints
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            b1, b2 = placed[i], placed[j]
            if b1["z"] + b1["dz"] - TOL <= b2["z"] or b2["z"] + b2["dz"] - TOL <= b1["z"]:
                continue
            if boxes_overlap(b1, b2):
                if support_of.get(b1["id"], -1) != b2["id"] and support_of.get(b2["id"], -1) != b1["id"]:
                    return False, f"Improper overlap between box id {b1['id']} and box id {b2['id']}"

    # (5d) load-bearing constraints
    total_load = {b["id"]: 0.0 for b in placed}
    for b in sorted(placed, key=lambda b: b["z"], reverse=True):
        load_here = b["weight"] + total_load[b["id"]]
        if b["id"] in support_of:
            total_load[support_of[b["id"]]] += load_here
    for b in placed:
        capacity = b["dx"] * b["dy"] * b["lb"]
        if total_load[b["id"]] > capacity + TOL:
            excess = total_load[b["id"]] - capacity
            return False, f"Load-bearing capacity exceeded for box id {b['id']}: overload {excess}"

    return True, None
'''
