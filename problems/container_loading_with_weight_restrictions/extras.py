"""Per-problem extras for CO-Bench Container Loading with Weight Restrictions.

Provides building-block tools for 3D container packing where each placement
must additionally respect per-box load-bearing capacity (weight stacked above
a box can't exceed dx * dy * lb, where lb is orientation-specific).

The container is axis-aligned, dimensions (L, W, H) in cm; coordinates are
(x, y, z) with z=vertical. A placement uses 1-indexed `box_type` plus
`orientation` in {1,2,3}:
  - 1: box length vertical -> (dx=width, dy=height, dz=length, lb=lb1)
  - 2: box width  vertical -> (dx=length, dy=height, dz=width,  lb=lb2)
  - 3: box height vertical -> (dx=length, dy=width,  dz=height, lb=lb3)
Each orientation is only legal if its `*_flag` is 1.

A placement is feasible iff every box (a) fits in the container, (b) does not
overlap any other except as an exact unique support stack, (c) every non-floor
box has exactly one supporting box whose top face fully contains the bottom
face, (d) the cumulative weight stacked on each supporting box never exceeds
its load-bearing capacity dx * dy * lb, (e) per-type usage <= count.

Tool groups:
  (1) Box / container queries:
        allowed_orientations(b), box_dims(b, orient), box_volume(b),
        box_weight(b), container_dims(), container_volume(),
        container_weight_capacity()
  (2) Placement queries (over a list of placement dicts):
        total_volume(placements), utilization(placements),
        total_weight(placements), is_within_weight(placements),
        compute_loads(placements)
  (3) Construction heuristics:
        greedy_pack_volume_first(),
        greedy_pack_density_aware(),
        greedy_pack_weight_aware()
  (Tier 4: 3D ILP omitted -- model too large to be useful at instance scale.)

All are exposed under tools[...] and described in EXTRA_TOOLS_DESCRIPTION.
"""
from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Any

TOL = 1e-6


def extra_tools(instance: dict) -> dict:
    """Factory: returns problem-specific tool callables given the instance."""
    container = instance["container"]              # (L, W, H) in cm
    container_L, container_W, container_H = container
    box_types = instance["box_types"]              # list[dict], length = n
    n_types = len(box_types)

    # Pre-decode the (dx, dy, dz, lb) tuple for every (type, orientation).
    # _ORIENT_DECODE[t][o] is None if orientation o is not allowed.
    _ORIENT_DECODE: List[List[Any]] = []
    for box in box_types:
        per_o: List[Any] = [None, None, None, None]  # 1..3
        # Orientation 1: length vertical
        if box["length_flag"] == 1:
            per_o[1] = (box["width"], box["height"], box["length"], box["lb1"])
        # Orientation 2: width vertical
        if box["width_flag"] == 1:
            per_o[2] = (box["length"], box["height"], box["width"], box["lb2"])
        # Orientation 3: height vertical
        if box["height_flag"] == 1:
            per_o[3] = (box["length"], box["width"], box["height"], box["lb3"])
        _ORIENT_DECODE.append(per_o)

    def _bt_index(box_type_1idx: int) -> int:
        i = int(box_type_1idx) - 1
        if i < 0 or i >= n_types:
            raise ValueError(
                f"box_type={box_type_1idx} out of range (1..{n_types})")
        return i

    # ==================================================================
    # (1) Box / container queries
    # ==================================================================
    def allowed_orientations(box_type_1idx: int) -> List[int]:
        """The list of orientations in {1,2,3} that are legal for this box."""
        i = _bt_index(box_type_1idx)
        return [o for o in (1, 2, 3) if _ORIENT_DECODE[i][o] is not None]

    def box_dims(box_type_1idx: int, orientation: int) -> Tuple[int, int, int]:
        """(dx, dy, dz) horizontal width, depth, and vertical height for one
        (box_type, orientation). Raises ValueError if orientation is illegal."""
        i = _bt_index(box_type_1idx)
        if orientation not in (1, 2, 3):
            raise ValueError(f"orientation must be in (1,2,3), got {orientation}")
        rec = _ORIENT_DECODE[i][orientation]
        if rec is None:
            raise ValueError(
                f"orientation {orientation} not allowed for box_type "
                f"{box_type_1idx}")
        return (rec[0], rec[1], rec[2])

    def box_volume(box_type_1idx: int) -> float:
        """Original (orientation-invariant) volume of a box of this type."""
        i = _bt_index(box_type_1idx)
        b = box_types[i]
        return float(b["length"]) * float(b["width"]) * float(b["height"])

    def box_weight(box_type_1idx: int) -> float:
        """Weight of a single box of this type (orientation-invariant)."""
        i = _bt_index(box_type_1idx)
        return float(box_types[i]["weight"])

    def container_dims() -> Tuple[int, int, int]:
        """Container (L, W, H) in cm."""
        return (int(container_L), int(container_W), int(container_H))

    def container_volume() -> float:
        """Container volume L*W*H in cm^3."""
        return float(container_L) * float(container_W) * float(container_H)

    def container_weight_capacity() -> float:
        """Total floor weight capacity = container_L * container_W * max(lb_*)
        upper bound: a single box sitting on the floor never sees a tighter
        floor-imposed limit than this. NOTE: the problem itself imposes no
        global weight cap; weight enters only via per-box load-bearing limits.
        This value is therefore a conservative ROOF on what is_within_weight()
        will accept across any feasible packing -- useful for early pruning."""
        # The floor itself has no lb; weight only enters via per-box lb.
        # The strongest per-(type,orient) lb (per unit area) bounds how much
        # weight can rest on any 1cm^2 floor cell. The floor area * max lb
        # gives a permissive upper bound on total stack weight.
        max_lb = 0.0
        for per_o in _ORIENT_DECODE:
            for rec in per_o:
                if rec is None:
                    continue
                if rec[3] > max_lb:
                    max_lb = float(rec[3])
        return float(container_L) * float(container_W) * max_lb

    # ==================================================================
    # (2) Placement queries
    # ==================================================================
    def _decoded(placements: Iterable[dict]) -> List[dict]:
        """Decode each placement into a record with (x,y,z,dx,dy,dz,lb,wt,vol).
        Silently skips placements with illegal box_type / orientation -- these
        are reported as 0 contribution rather than crashing. Use tools[
        'is_feasible'] for strict validation."""
        out: List[dict] = []
        for idx, p in enumerate(placements):
            try:
                i = _bt_index(p["box_type"])
                rec = _ORIENT_DECODE[i][int(p["orientation"])]
                if rec is None:
                    continue
                dx, dy, dz, lb = rec
                b = box_types[i]
                out.append({
                    "id": idx,
                    "type_idx": i,
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "z": float(p["z"]),
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "lb": float(lb),
                    "weight": float(b["weight"]),
                    "volume": float(b["length"]) * float(b["width"]) * float(b["height"]),
                })
            except Exception:
                continue
        return out

    def total_volume(placements: Iterable[dict]) -> float:
        """Sum of original box volumes across `placements` (cm^3). Skips
        placements whose box_type/orientation are not decodable. Does NOT
        check overlap / support / weight."""
        return sum(d["volume"] for d in _decoded(placements))

    def utilization(placements: Iterable[dict]) -> float:
        """total_volume(placements) / container_volume(). NOT a feasibility
        check; use tools['is_feasible'] for that."""
        cv = container_volume()
        if cv <= 0:
            return 0.0
        return total_volume(placements) / cv

    def total_weight(placements: Iterable[dict]) -> float:
        """Sum of weights of placed boxes (orientation-invariant). Skips
        undecodable placements."""
        return sum(d["weight"] for d in _decoded(placements))

    def compute_loads(placements: Iterable[dict]) -> Dict[int, float]:
        """For each placement index, return the cumulative weight resting on
        TOP of it (excluding its own weight). Determined by walking the
        unique-support chain top-down. If the support relation is ambiguous
        or missing for some non-floor box, that box contributes 0 to its
        (would-be) supporter -- treat as undefined and rely on
        tools['is_feasible'] for the strict check.

        Returns: dict mapping placement index -> load resting on top (float).
        """
        placed = _decoded(placements)
        # Find unique supporter for each non-floor box (mirrors eval_func).
        support_of: Dict[int, int] = {}
        for b in placed:
            if abs(b["z"]) < TOL:
                continue
            cand = None
            ambiguous = False
            for o in placed:
                if o["id"] == b["id"]:
                    continue
                if abs(o["z"] + o["dz"] - b["z"]) > TOL:
                    continue
                # b horizontally fully inside o's top face
                if b["x"] + TOL < o["x"]:
                    continue
                if b["x"] + b["dx"] - TOL > o["x"] + o["dx"]:
                    continue
                if b["y"] + TOL < o["y"]:
                    continue
                if b["y"] + b["dy"] - TOL > o["y"] + o["dy"]:
                    continue
                if cand is not None:
                    ambiguous = True
                    break
                cand = o
            if cand is not None and not ambiguous:
                support_of[b["id"]] = cand["id"]
        # Tally load top-down.
        load_on_top: Dict[int, float] = {b["id"]: 0.0 for b in placed}
        for b in sorted(placed, key=lambda x: x["z"], reverse=True):
            stacked_here = b["weight"] + load_on_top[b["id"]]
            if b["id"] in support_of:
                load_on_top[support_of[b["id"]]] += stacked_here
        return load_on_top

    def is_within_weight(placements: Iterable[dict]) -> bool:
        """True iff every placed box's load-bearing capacity (dx*dy*lb) is
        not exceeded by the weight resting on it (computed via the same
        unique-support chain as eval_func). Returns False if any box's load
        exceeds capacity. Does NOT check geometric feasibility -- if support
        relations are ambiguous / missing, weights propagate as 0 and this
        may report True erroneously; use tools['is_feasible'] for the strict
        combined check."""
        placed = _decoded(placements)
        loads = compute_loads(placements)
        for b in placed:
            cap = b["dx"] * b["dy"] * b["lb"]
            if loads.get(b["id"], 0.0) > cap + TOL:
                return False
        return True

    # ==================================================================
    # (3) Construction heuristics
    # ==================================================================
    def _best_floor_orient(box_idx: int, score_fn) -> Tuple[int, Tuple[int, int, int], float]:
        """Pick the legal orientation maximizing `score_fn(dx, dy, dz, lb)`.
        Returns (orient, (dx,dy,dz), lb). Raises ValueError if no orient legal."""
        best = None
        for o in (1, 2, 3):
            rec = _ORIENT_DECODE[box_idx][o]
            if rec is None:
                continue
            dx, dy, dz, lb = rec
            if dx > container_L + TOL or dy > container_W + TOL or dz > container_H + TOL:
                continue
            s = score_fn(dx, dy, dz, lb)
            if best is None or s > best[3]:
                best = (o, (dx, dy, dz), float(lb), s)
        if best is None:
            raise ValueError(f"no legal floor orientation for box type {box_idx + 1}")
        return best[0], best[1], best[2]

    def _layer_pack(score_fn) -> List[dict]:
        """Generic layer-by-layer floor packer using a 2D shelf heuristic.

        For each box type (in the order yielded by `score_fn`-sorted list),
        pick the orientation maximizing score_fn, then place copies in shelf
        rows on the floor (z=0) only -- so no stacking, so support / weight
        constraints are trivially satisfied. Returns the list of placement
        dicts.

        This intentionally trades stack-height for simplicity: a layer-only
        packing always has 0 load on every box, so it cannot violate weight.
        Use it as a safe warm start, then have the LLM add stack layers."""
        # Build (type_idx, count, score) and sort by score desc.
        ranked = []
        for i, box in enumerate(box_types):
            # Use orientation 3 dimensions if available else first allowed for
            # the ranking pass; score_fn is called on the chosen-best orient.
            try:
                _, (dx, dy, dz), lb = _best_floor_orient(i, score_fn)
            except ValueError:
                continue
            ranked.append((score_fn(dx, dy, dz, lb), i, box["count"]))
        ranked.sort(reverse=True)

        placements: List[dict] = []
        usage = [0] * n_types
        # Single floor layer using shelf algorithm:
        #   y advances along container_W in "shelves",
        #   each shelf packs boxes left-to-right along container_L,
        #   shelf depth = max dy of boxes in that shelf,
        #   z = 0 (floor only).
        cur_y = 0.0
        shelf_depth = 0.0
        cur_x = 0.0
        for _, i, count in ranked:
            if count <= 0:
                continue
            try:
                orient, (dx, dy, dz), lb = _best_floor_orient(i, score_fn)
            except ValueError:
                continue
            # Place as many as possible.
            while usage[i] < count:
                # Need to start a new shelf if this box doesn't fit current.
                if cur_x + dx > container_L + TOL:
                    cur_y += shelf_depth
                    cur_x = 0.0
                    shelf_depth = 0.0
                if cur_y + dy > container_W + TOL:
                    return placements  # floor is full
                placements.append({
                    "box_type": i + 1,
                    "orientation": orient,
                    "x": float(cur_x),
                    "y": float(cur_y),
                    "z": 0.0,
                })
                usage[i] += 1
                cur_x += dx
                if dy > shelf_depth:
                    shelf_depth = dy
        return placements

    def greedy_pack_volume_first() -> List[dict]:
        """Floor-only shelf packer that prefers boxes whose (chosen
        orientation's) footprint is largest. Stacks are not built, so weight
        constraints are vacuously satisfied. Returns list[placement_dict]
        suitable for the solution's 'placements' key."""
        return _layer_pack(lambda dx, dy, dz, lb: dx * dy * dz)

    def greedy_pack_density_aware() -> List[dict]:
        """Floor-only shelf packer that prefers boxes with high
        volume-per-weight ratio first (light heavy-volume items get prime
        floor real estate). Avoids weight bottlenecks even though this packer
        never stacks; useful as a starting solution that can be later
        augmented with stacks of lighter boxes."""
        def score(dx, dy, dz, lb):
            # volume / (weight + tiny) approximated via dz proxy is not
            # available here; use volume itself and break ties on lb (higher
            # load-bearing => better candidate to support future stack).
            return (dx * dy * dz, lb)
        # _layer_pack expects a scalar; convert tuple via lexicographic to
        # scalar by pairing primary*K + secondary.
        K = 1e6
        return _layer_pack(lambda dx, dy, dz, lb: (dx * dy * dz) * K + lb)

    def greedy_pack_weight_aware() -> List[dict]:
        """Floor-only shelf packer that prefers boxes with the highest load-
        bearing capacity per unit footprint (dx*dy*lb) first -- those are the
        best candidates to support stacks above them, so placing them on the
        floor leaves the most room for the LLM to add stack layers on top
        without overloading. Like the other layer packers, it never stacks,
        so it returns a guaranteed weight-feasible warm start."""
        return _layer_pack(lambda dx, dy, dz, lb: dx * dy * lb + dx * dy * dz * 1e-9)

    # ==================================================================
    # (4) Column-stacking strong solver
    # ==================================================================
    def max_column_height(box_type_1idx: int, orientation: int) -> int:
        """Maximum number of identical boxes (same type + orientation) that
        can be stacked in a single vertical column from z=0 upward,
        respecting (a) container height, (b) load-bearing capacity of every
        box in the stack, (c) per-type count. Returns 0 if the orientation
        is illegal or the box does not fit at all.

        Column physics: if k boxes are stacked, the bottom box bears
        (k-1) * weight of its supporters' contents above. Its capacity is
        dx * dy * lb. So we need (k-1) * weight <= dx * dy * lb (every
        higher box has strictly less load on it, so checking the bottom
        is sufficient). Plus k * dz <= H. Plus k <= count."""
        i = _bt_index(box_type_1idx)
        if orientation not in (1, 2, 3):
            return 0
        rec = _ORIENT_DECODE[i][orientation]
        if rec is None:
            return 0
        dx, dy, dz, lb = rec
        if dx > container_L + TOL or dy > container_W + TOL or dz > container_H + TOL:
            return 0
        w = float(box_types[i]["weight"])
        if w <= 0:
            k_by_weight = 10**9
        else:
            capacity_per_box = dx * dy * lb
            k_by_weight = int(capacity_per_box / w) + 1
        k_by_height = int((container_L + TOL) / dz) if False else int((container_H + TOL) / dz)
        k_by_count = int(box_types[i]["count"])
        return max(0, min(k_by_weight, k_by_height, k_by_count))

    def _column_score(i: int, orient: int) -> tuple:
        """Score a (box_type_0idx, orientation) candidate column: returns
        (-volume_filled, -base_area). Higher volume_filled comes first;
        ties broken by larger base area (preserves later flexibility)."""
        rec = _ORIENT_DECODE[i][orient]
        if rec is None:
            return (0.0, 0.0)
        dx, dy, dz, _lb = rec
        k = max_column_height(i + 1, orient)
        if k <= 0:
            return (0.0, 0.0)
        vol = dx * dy * dz * k
        return (-vol, -dx * dy)

    def solve_column_stacked() -> list:
        """STRONG GREEDY PACKER. Builds the packing as a set of vertical
        columns; each column is a vertical stack of identical boxes (same
        type + orientation) so the support and load-bearing constraints
        are AUTOMATICALLY satisfied (each box exactly supports the box
        directly on top of it; column physics caps the height).

        Strategy:
          1. For each (box_type, orientation) compute the column physics:
             maximum column height k under container_H + load + count.
          2. Rank candidate columns by total filled volume (descending).
          3. Shelf-pack the (dx, dy) column footprints on the floor of
             the container using a next-fit-decreasing-height shelf
             algorithm (shelves run along x; each shelf opens with the
             first column's dy as its depth).
          4. For each placed column, emit k box placements at (x, y, z)
             where z = j * dz for j = 0..k-1.

        Returns a list of placement dicts ready to drop into the solution.
        Always returns a FEASIBLE packing (support + load + count + bounds
        all guaranteed by construction)."""
        # Build sorted candidate column list.
        cands = []
        usage = [0] * n_types
        for i in range(n_types):
            for o in (1, 2, 3):
                if _ORIENT_DECODE[i][o] is None:
                    continue
                k0 = max_column_height(i + 1, o)
                if k0 <= 0:
                    continue
                cands.append((_column_score(i, o), i, o))
        cands.sort()  # ascending of (-vol, -base_area) == descending vol

        placements: list[dict] = []
        # Shelf state.
        cur_y = 0.0
        shelf_depth = 0.0
        cur_x = 0.0
        for _score, i, o in cands:
            # How many columns of this (i, o) are still available?
            remaining_count = int(box_types[i]["count"]) - usage[i]
            if remaining_count <= 0:
                continue
            rec = _ORIENT_DECODE[i][o]
            dx, dy, dz, _lb = rec
            # Greedily try to place as many columns as possible.
            placed_in_pass = True
            while placed_in_pass and remaining_count > 0:
                placed_in_pass = False
                # Per-column height: limited by remaining count.
                k_max = max_column_height(i + 1, o)
                k = min(k_max, remaining_count)
                if k <= 0:
                    break
                # Try current shelf.
                if cur_x + dx > container_L + TOL:
                    # New shelf.
                    cur_y += shelf_depth
                    cur_x = 0.0
                    shelf_depth = 0.0
                if cur_y + dy > container_W + TOL:
                    # Floor is full.
                    return placements
                # Place column at (cur_x, cur_y).
                for j in range(k):
                    placements.append({
                        "box_type": i + 1,
                        "orientation": o,
                        "x": float(cur_x),
                        "y": float(cur_y),
                        "z": float(j * dz),
                    })
                usage[i] += k
                remaining_count -= k
                cur_x += dx
                if dy > shelf_depth:
                    shelf_depth = dy
                placed_in_pass = True
        return placements

    # ==================================================================
    # (5) Solution-dict builder + one-shot solver
    # ==================================================================
    def make_solution(placements) -> dict:
        """Wrap a list of placement dicts into the EXACT dict shape eval_func
        expects: {'instance': 1, 'util': float, 'm': int,
                  'placements': list[{'box_type','orientation','x','y','z'}]}.
        Computes util = total_volume(placements) / container_volume()."""
        pl = list(placements) if placements else []
        cv = container_volume()
        u = (total_volume(pl) / cv) if cv > 0 else 0.0
        return {
            "instance": 1,
            "util": float(u),
            "m": len(pl),
            "placements": pl,
        }

    def solve_default(time_limit_s: float = 10.0) -> dict:
        """ONE-SHOT STRONG SOLVER. Returns the complete solution dict ready
        to return directly. Runs solve_column_stacked (column-based packer
        that satisfies support + load + count + bounds by construction)
        and wraps the result. ONE LINE:
            return tools['solve_default'](time_limit_s=10)
        """
        pl = solve_column_stacked()
        # Compare against the floor-only baselines as a sanity net.
        floor = greedy_pack_volume_first()
        if total_volume(floor) > total_volume(pl):
            pl = floor
        return make_solution(pl)

    return {
        # (5) one-shot + builder (CALL FIRST)
        "solve_default": solve_default,
        "make_solution": make_solution,
        # (4) column-stacking strong solver
        "solve_column_stacked": solve_column_stacked,
        "max_column_height": max_column_height,
        # (3) Construction (floor-only fallbacks)
        "greedy_pack_volume_first": greedy_pack_volume_first,
        "greedy_pack_density_aware": greedy_pack_density_aware,
        "greedy_pack_weight_aware": greedy_pack_weight_aware,
        # (2) Placement-list queries
        "total_volume": total_volume,
        "utilization": utilization,
        "total_weight": total_weight,
        "compute_loads": compute_loads,
        "is_within_weight": is_within_weight,
        # (1) Queries
        "allowed_orientations": allowed_orientations,
        "box_dims": box_dims,
        "box_volume": box_volume,
        "box_weight": box_weight,
        "container_dims": container_dims,
        "container_volume": container_volume,
        "container_weight_capacity": container_weight_capacity,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (5) ONE-SHOT STRONG SOLVER (call this first!) -----
    {
        "name": "solve_default",
        "input": "time_limit_s: float = 10.0",
        "output": "dict {'instance', 'util', 'm', 'placements'}",
        "purpose": (
            "RECOMMENDED START: returns a complete solution dict ready to return "
            "directly. Builds the packing as a set of vertical columns (each "
            "column is a stack of identical boxes), so support + load-bearing + "
            "count + bounds constraints are SATISFIED BY CONSTRUCTION. ONE LINE: "
            "`return tools['solve_default'](time_limit_s=10)`."
        ),
    },
    {
        "name": "make_solution",
        "input": "placements: Iterable[dict]",
        "output": "dict {'instance', 'util', 'm', 'placements'}",
        "purpose": (
            "Build the EXACT solution dict shape eval_func wants: "
            "{'instance': 1, 'util': float, 'm': len(placements), "
            "'placements': list}. Use on the output of "
            "solve_column_stacked() / greedy_pack_*() so you never return "
            "the wrong dict shape."
        ),
    },
    # ----- (4) Column-stacking strong tools -----
    {
        "name": "solve_column_stacked",
        "input": "(no args)",
        "output": "list[placement_dict]",
        "purpose": (
            "Strong greedy packer that builds the packing as vertical columns "
            "of identical (type + orientation) boxes. Because every column has "
            "constant (dx, dy) along its height, the support constraint is "
            "AUTOMATICALLY satisfied (each box is supported by an identical box "
            "below). Load-bearing is enforced via max_column_height. Floor is "
            "shelf-packed (rows along x, depth = column's dy). Returns a "
            "feasible placements list."
        ),
    },
    {
        "name": "max_column_height",
        "input": "box_type_1idx: int, orientation: int",
        "output": "int",
        "purpose": (
            "Max number of identical boxes that can stack in a single column "
            "(z=0 upward) under (a) container_H (b) load-bearing capacity "
            "(bottom box bears (k-1)*weight, must be <= dx*dy*lb) (c) per-type "
            "count. Returns 0 if the orientation is illegal or the box does "
            "not fit at all."
        ),
    },
    # ----- (1) Queries -----
    {
        "name": "allowed_orientations",
        "input": "box_type_1idx: int",
        "output": "list[int]",
        "purpose": (
            "Return the legal orientations in {1,2,3} for this box type. "
            "Orientation o is legal iff the underlying box's o-th *_flag is 1. "
            "Always consult this before constructing a placement."
        ),
    },
    {
        "name": "box_dims",
        "input": "box_type_1idx: int, orientation: int",
        "output": "(dx, dy, dz) tuple of int",
        "purpose": (
            "(dx, dy) horizontal footprint and dz vertical extent of one box "
            "in a given orientation. Raises if orientation is illegal."
        ),
    },
    {
        "name": "box_volume",
        "input": "box_type_1idx: int",
        "output": "float",
        "purpose": "Orientation-invariant volume length*width*height of a box of this type.",
    },
    {
        "name": "box_weight",
        "input": "box_type_1idx: int",
        "output": "float",
        "purpose": (
            "Weight of one box of this type (orientation-invariant). Drives "
            "load-bearing feasibility -- a stack's accumulated weight resting "
            "on box B may not exceed B.dx * B.dy * B.lb."
        ),
    },
    {
        "name": "container_dims",
        "input": "(no args)",
        "output": "(L, W, H) tuple of int",
        "purpose": "Container interior dimensions in cm.",
    },
    {
        "name": "container_volume",
        "input": "(no args)",
        "output": "float",
        "purpose": "Container interior volume in cm^3. Denominator of utilization.",
    },
    {
        "name": "container_weight_capacity",
        "input": "(no args)",
        "output": "float",
        "purpose": (
            "Conservative upper bound on total stack weight any feasible "
            "packing can carry: floor_area * max(lb) across all (type, "
            "orientation). The problem has no explicit global cap, but this "
            "bound is useful for early pruning when picking candidate boxes."
        ),
    },
    # ----- (2) Placement-list queries -----
    {
        "name": "total_volume",
        "input": "placements: list[dict]",
        "output": "float",
        "purpose": (
            "Sum of placed-box volumes (cm^3). Skips placements with illegal "
            "box_type / orientation -- call tools['is_feasible'] for strict "
            "feasibility, not this."
        ),
    },
    {
        "name": "utilization",
        "input": "placements: list[dict]",
        "output": "float",
        "purpose": (
            "total_volume(placements) / container_volume(). Same value the "
            "problem maximizes when the solution is feasible (else the actual "
            "score is 0). NOT a feasibility check."
        ),
    },
    {
        "name": "total_weight",
        "input": "placements: list[dict]",
        "output": "float",
        "purpose": "Sum of weights of placed boxes. Skips undecodable placements.",
    },
    {
        "name": "compute_loads",
        "input": "placements: list[dict]",
        "output": "dict[int, float]",
        "purpose": (
            "For each placement index, the weight currently resting ON TOP "
            "of that box (not counting its own weight). Computed via the same "
            "unique-support chain that eval_func uses. If support is "
            "ambiguous / missing for a non-floor box, that box's load "
            "contribution propagates as 0 -- use only as an estimate; rely on "
            "tools['is_feasible'] for the authoritative check."
        ),
    },
    {
        "name": "is_within_weight",
        "input": "placements: list[dict]",
        "output": "bool",
        "purpose": (
            "True iff no box's load-bearing capacity dx*dy*lb is exceeded by "
            "the weight resting on top of it. Geometric feasibility is NOT "
            "checked -- combine with tools['is_feasible'] for full check, or "
            "use this as a cheap weight-only filter inside local search."
        ),
    },
    # ----- (3) Construction -----
    {
        "name": "greedy_pack_volume_first",
        "input": "(no args)",
        "output": "list[placement_dict]",
        "purpose": (
            "Floor-only shelf packer that places boxes in descending volume "
            "(per chosen-orientation footprint*height) order. Because it "
            "never stacks, the result trivially satisfies support and "
            "load-bearing constraints -- a safe weight-feasible warm start. "
            "Wrap as {'instance': 1, 'util': ..., 'm': len, 'placements': "
            "<this list>}."
        ),
    },
    {
        "name": "greedy_pack_density_aware",
        "input": "(no args)",
        "output": "list[placement_dict]",
        "purpose": (
            "Floor-only shelf packer that prefers boxes with large volume, "
            "tie-breaking on higher load-bearing capacity (so the boxes left "
            "on the floor are the best supporters for future stack layers). "
            "Weight-feasible by construction (no stacking)."
        ),
    },
    {
        "name": "greedy_pack_weight_aware",
        "input": "(no args)",
        "output": "list[placement_dict]",
        "purpose": (
            "Floor-only shelf packer that prioritizes boxes with the largest "
            "dx*dy*lb -- i.e. those most capable of supporting future stacks "
            "above them. Use as a base, then have the LLM add stack layers "
            "of lighter / smaller boxes on top while keeping is_within_weight "
            "true. Weight-feasible by construction."
        ),
    },
]
