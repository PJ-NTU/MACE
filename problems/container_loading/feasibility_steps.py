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
    if not isinstance(placements, (list, tuple)):
        return False, f"'placements' must be list, got {type(placements).__name__}"

    required_keys = {"box_type", "container_id", "x", "y", "z", "v", "hswap"}

    cont_len, cont_wid, cont_ht = container

    used_counts = {}
    placements_by_container = {}

    # (4) per-element value constraints
    for idx, pmt in enumerate(placements):
        if not isinstance(pmt, dict):
            return False, f"placement[{idx}] must be dict, got {type(pmt).__name__}"
        missing = required_keys - pmt.keys()
        if missing:
            return False, f"placement[{idx}] missing keys: {missing}"

        bt = pmt["box_type"]
        if bt not in box_types:
            return False, f"placement[{idx}] has unknown box_type {bt}"

        info = box_types[bt]
        dims = info["dims"]
        flags = info["flags"]

        v = pmt["v"]
        if v not in [0, 1, 2]:
            return False, f"placement[{idx}] 'v' must be 0, 1, or 2, got {v}"
        if flags[v] != 1:
            return False, f"placement[{idx}] vertical dimension {v} not allowed for box_type {bt}"

        hswap = pmt["hswap"]
        if hswap not in [0, 1]:
            return False, f"placement[{idx}] 'hswap' must be 0 or 1, got {hswap}"

        horz_idx = [i for i in [0, 1, 2] if i != v]
        h1 = dims[horz_idx[0]]
        h2 = dims[horz_idx[1]]
        if hswap == 1:
            h1, h2 = h2, h1
        vert = dims[v]

        x, y, z = pmt["x"], pmt["y"], pmt["z"]
        if x < 0 or y < 0 or z < 0:
            return False, f"placement[{idx}] has negative coordinate (x={x}, y={y}, z={z})"
        if x + h1 > cont_len or y + h2 > cont_wid or z + vert > cont_ht:
            return False, (f"placement[{idx}] exceeds container bounds: "
                           f"box end ({x+h1},{y+h2},{z+vert}) > container ({cont_len},{cont_wid},{cont_ht})")

        pmt["_odims"] = (h1, h2, vert)
        pmt["_opos"] = (x, y, z)

        used_counts[bt] = used_counts.get(bt, 0) + 1
        cid = pmt["container_id"]
        placements_by_container.setdefault(cid, []).append(pmt)

    # (5) cross-element / global constraints
    for bt, cnt in used_counts.items():
        if cnt > box_types[bt]["count"]:
            return False, f"box_type {bt} used {cnt} times but only {box_types[bt]['count']} available"

    for cid, plist in placements_by_container.items():
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                x1, y1, z1 = plist[i]["_opos"]
                w1, d1, h1 = plist[i]["_odims"]
                x2, y2, z2 = plist[j]["_opos"]
                w2, d2, h2 = plist[j]["_odims"]
                overlap = not (x1+w1 <= x2 or x2+w2 <= x1 or
                               y1+d1 <= y2 or y2+d2 <= y1 or
                               z1+h1 <= z2 or z2+h2 <= z1)
                if overlap:
                    return False, f"placements {i} and {j} in container {cid} overlap"

    return True, None
'''
