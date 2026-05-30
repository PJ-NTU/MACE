"""Per-problem extras for CO-Bench Container Loading (3D bin packing).

Provides primitive building blocks so the LLM can compose 3D packing
heuristics (wall-building, corner / extreme-point placement, swap-based
local search) without re-deriving every overlap test or orientation
enumeration from scratch.

Tools fall in 4 tiers (mirrors the bin_packing_one_dimensional layout):

  (1) Queries:        container_dims, box_dims, box_value (== volume),
                      box_count_available, n_box_types, box_orientations
  (2) Feasibility:    overlap_3d, used_volume, used_count, fits_in_container,
                      utilization
  (3) Construction:   wall_building_pack (Bischoff-Ratcliff style),
                      corner_pack_3d (extreme-point), try_place_at_corner_3d
  (4) Local search:   apply_swap_boxes  (perturb-and-repack)

CO-Bench solution schema:
    {"placements": [{"box_type": int, "container_id": int,
                     "x": int, "y": int, "z": int,
                     "v": int (0/1/2), "hswap": int (0/1)}, ...]}

The "container_id" is always 0 in this task (a single container per
instance), but we honor whatever value the LLM emits.

Note on "value": this task is a pure volume-utilization problem (no
per-box weight). `box_value` returns box VOLUME, which is the right
quantity to greedy-sort on for both density-first and largest-first
strategies. If you want raw count, use len(...) on box_count_available
results.

All construction tools return a list of placement dicts directly
compatible with the CO-Bench solution schema.
"""
from __future__ import annotations

import random
import time
from typing import Iterable, Optional


def extra_tools(instance: dict) -> dict:
    """Factory: returns Container-Loading tool callables for the instance.

    Instance schema (from CO-Bench load_data):
      - problem_index: int
      - container:     tuple(L, W, H)        (int)
      - box_types:     dict[int -> {'dims': [d1,d2,d3],
                                    'flags':[f1,f2,f3],
                                    'count': int}]
    """
    L, W, H = (int(v) for v in instance["container"])
    container_volume = L * W * H
    box_types_raw: dict = instance["box_types"]
    # Stable order for box types (sorted by id).
    type_ids: list[int] = sorted(box_types_raw.keys())

    # Pre-extract per-type info for fast access.
    # _dims[bt] = (d0, d1, d2); _flags[bt] = (f0, f1, f2); _count[bt] = int
    _dims: dict[int, tuple[int, int, int]] = {}
    _flags: dict[int, tuple[int, int, int]] = {}
    _count: dict[int, int] = {}
    for bt in type_ids:
        info = box_types_raw[bt]
        d = info["dims"]
        f = info["flags"]
        _dims[bt] = (int(d[0]), int(d[1]), int(d[2]))
        _flags[bt] = (int(f[0]), int(f[1]), int(f[2]))
        _count[bt] = int(info["count"])

    # Pre-enumerate legal (v, hswap) orientations per box type.
    # Each entry is (v, hswap, h1, h2, vert) with oriented physical sizes.
    def _enum_orientations(bt: int) -> list[tuple[int, int, int, int, int]]:
        d = _dims[bt]
        f = _flags[bt]
        out: list[tuple[int, int, int, int, int]] = []
        for v in (0, 1, 2):
            if f[v] != 1:
                continue
            horz_idx = [i for i in (0, 1, 2) if i != v]
            h1_base = d[horz_idx[0]]
            h2_base = d[horz_idx[1]]
            vert = d[v]
            # hswap = 0
            out.append((v, 0, h1_base, h2_base, vert))
            # hswap = 1 (only meaningful if it gives a different box)
            if h1_base != h2_base:
                out.append((v, 1, h2_base, h1_base, vert))
        return out

    _orient_cache: dict[int, list[tuple[int, int, int, int, int]]] = {
        bt: _enum_orientations(bt) for bt in type_ids
    }

    # ==================================================================
    # Helpers (closure-private)
    # ==================================================================
    def _oriented_dims(bt: int, v: int, hswap: int) -> tuple[int, int, int]:
        """Return (size_x, size_y, size_z) for the given (bt, v, hswap)."""
        d = _dims[bt]
        horz_idx = [i for i in (0, 1, 2) if i != v]
        h1 = d[horz_idx[0]]
        h2 = d[horz_idx[1]]
        if hswap == 1:
            h1, h2 = h2, h1
        vert = d[v]
        return h1, h2, vert

    def _boxes_overlap(p1: tuple[int, int, int], s1: tuple[int, int, int],
                       p2: tuple[int, int, int], s2: tuple[int, int, int]) -> bool:
        if p1[0] + s1[0] <= p2[0] or p2[0] + s2[0] <= p1[0]:
            return False
        if p1[1] + s1[1] <= p2[1] or p2[1] + s2[1] <= p1[1]:
            return False
        if p1[2] + s1[2] <= p2[2] or p2[2] + s2[2] <= p1[2]:
            return False
        return True

    def _placement_geometry(pmt: dict) -> tuple[
            tuple[int, int, int], tuple[int, int, int]]:
        """(pos, dims) for a placement dict. Trusts that bt/v/hswap are valid."""
        bt = pmt["box_type"]
        v = pmt["v"]
        hswap = pmt["hswap"]
        return ((int(pmt["x"]), int(pmt["y"]), int(pmt["z"])),
                _oriented_dims(bt, v, hswap))

    def _gather_geom(placements: Iterable[dict]):
        """Yield (pos, dims) for each placement -- precomputed for inner loops."""
        return [_placement_geometry(p) for p in placements]

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def container_dims() -> tuple[int, int, int]:
        """Container dimensions (L, W, H) = (length-x, width-y, height-z)."""
        return (L, W, H)

    def box_dims(bt: int) -> tuple[int, int, int]:
        """Raw (unoriented) (d1, d2, d3) of box type `bt`."""
        if bt not in _dims:
            raise KeyError(f"box_type {bt} not in instance")
        return _dims[bt]

    def box_value(bt: int) -> int:
        """Volume (d1*d2*d3) of one box of type `bt`. This problem has no
        separate per-box value -- volume IS the value, since the objective
        is volume utilization. Use as a sort key for greedy heuristics."""
        if bt not in _dims:
            raise KeyError(f"box_type {bt} not in instance")
        d = _dims[bt]
        return d[0] * d[1] * d[2]

    def box_count_available(bt: int) -> int:
        """Number of boxes of type `bt` available to place."""
        if bt not in _count:
            raise KeyError(f"box_type {bt} not in instance")
        return _count[bt]

    def n_box_types() -> int:
        """Number of distinct box types in this instance."""
        return len(type_ids)

    def box_orientations(bt: int) -> list[tuple[int, int, int, int, int]]:
        """All LEGAL orientations of box type `bt`, honoring vertical-flag
        constraints. Each tuple is (v, hswap, size_x, size_y, size_z) where
        (v, hswap) are the fields needed in the solution dict and the three
        sizes are the resulting oriented dimensions. Useful when scanning
        which orientation packs into a given empty corner."""
        if bt not in _orient_cache:
            raise KeyError(f"box_type {bt} not in instance")
        return list(_orient_cache[bt])

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def overlap_3d(placements: Iterable[dict], candidate: dict) -> bool:
        """True iff `candidate` (a placement dict) overlaps ANY box already
        in `placements`. O(len(placements)). Touching faces are NOT overlap
        (matches CO-Bench's eval_func). Both `candidate` and entries in
        `placements` must carry valid (box_type, x, y, z, v, hswap)."""
        cpos, cdims = _placement_geometry(candidate)
        for p in placements:
            pos, dims = _placement_geometry(p)
            if _boxes_overlap(cpos, cdims, pos, dims):
                return True
        return False

    def fits_in_container(candidate: dict) -> bool:
        """True iff `candidate` lies entirely within the container box
        [0,L) x [0,W) x [0,H), has a legal vertical orientation, and has
        nonneg coordinates. Does NOT check overlap with other placements."""
        bt = candidate.get("box_type")
        if bt not in _dims:
            return False
        v = candidate.get("v")
        if v not in (0, 1, 2) or _flags[bt][v] != 1:
            return False
        hswap = candidate.get("hswap")
        if hswap not in (0, 1):
            return False
        x = int(candidate.get("x", -1))
        y = int(candidate.get("y", -1))
        z = int(candidate.get("z", -1))
        if x < 0 or y < 0 or z < 0:
            return False
        sx, sy, sz = _oriented_dims(bt, v, hswap)
        if x + sx > L or y + sy > W or z + sz > H:
            return False
        return True

    def used_volume(placements: Iterable[dict]) -> int:
        """Sum of oriented volumes for the given placements. Does NOT check
        feasibility -- if you call it on overlapping placements you still
        get the raw sum. O(n)."""
        total = 0
        for p in placements:
            _, dims = _placement_geometry(p)
            total += dims[0] * dims[1] * dims[2]
        return total

    def used_count(placements: Iterable[dict], box_type: int) -> int:
        """How many boxes of `box_type` appear in `placements`. Useful to
        check against box_count_available(bt) before adding another."""
        return sum(1 for p in placements if p.get("box_type") == box_type)

    def utilization(placements: Iterable[dict]) -> float:
        """used_volume / container_volume. Matches CO-Bench's objective on
        the assumption that `placements` is feasible (no overlap, in-bounds,
        within counts). Useful as a fast local proxy for tools['objective']."""
        if container_volume <= 0:
            return 0.0
        return used_volume(placements) / container_volume

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def try_place_at_corner_3d(
        placements: list[dict],
        box_type: int,
        orientations: Optional[Iterable[tuple[int, int]]] = None,
    ) -> Optional[dict]:
        """Try to add one box of `box_type` at some feasible extreme point
        (corner) of the current packing. Returns a new placement dict
        (NOT mutated into `placements`) or None if nothing fits.

        Strategy:
          * Candidate positions are EXTREME POINTS: (0,0,0) plus the three
            face-front corners of every already-placed box, i.e.
            (x+sx, y, z), (x, y+sy, z), (x, y, z+sz). This is a standard
            efficient cover of feasible left-back-bottom anchors.
          * For each candidate position, scan orientations in the given
            order (default: cached legal orientations for this box type)
            and accept the FIRST that fits in the container and does not
            overlap. The result is biased toward earlier orientations.
          * Among feasible (pos, orientation) pairs, picks the one with
            the smallest (z, y, x) anchor (bottom-back-left first).

        orientations may be passed as a list of (v, hswap) pairs to
        restrict / reorder the search (e.g., force a specific rotation
        choice). If None, uses all legal orientations.

        Returns a placement dict with container_id=0 on success."""
        if box_type not in _orient_cache:
            return None
        cand_orients: list[tuple[int, int, int, int, int]]
        if orientations is None:
            cand_orients = _orient_cache[box_type]
        else:
            wanted = set((int(v), int(h)) for v, h in orientations)
            cand_orients = [o for o in _orient_cache[box_type]
                            if (o[0], o[1]) in wanted]
        if not cand_orients:
            return None
        # Build candidate anchor positions.
        anchors: set[tuple[int, int, int]] = {(0, 0, 0)}
        for p in placements:
            pos, dims = _placement_geometry(p)
            x, y, z = pos
            sx, sy, sz = dims
            if x + sx < L:
                anchors.add((x + sx, y, z))
            if y + sy < W:
                anchors.add((x, y + sy, z))
            if z + sz < H:
                anchors.add((x, y, z + sz))
        # Sort anchors bottom-back-left first.
        sorted_anchors = sorted(anchors, key=lambda a: (a[2], a[1], a[0]))
        placed_geom = _gather_geom(placements)
        for (ax, ay, az) in sorted_anchors:
            for (v, hswap, sx, sy, sz) in cand_orients:
                if ax + sx > L or ay + sy > W or az + sz > H:
                    continue
                ok = True
                for (pos, dims) in placed_geom:
                    if _boxes_overlap((ax, ay, az), (sx, sy, sz), pos, dims):
                        ok = False
                        break
                if ok:
                    return {
                        "box_type": int(box_type),
                        "container_id": 0,
                        "x": int(ax),
                        "y": int(ay),
                        "z": int(az),
                        "v": int(v),
                        "hswap": int(hswap),
                    }
        return None

    def corner_pack_3d(
        box_order: Optional[Iterable[int]] = None,
        allow_rotation: bool = True,
    ) -> list[dict]:
        """Greedy extreme-point packer. Processes `box_order` (a sequence of
        box-type ids, possibly with repeats) and for each box calls
        try_place_at_corner_3d. Skips a box if no corner fits, never aborts.

        Default `box_order`: every box of every type, repeated by available
        count, sorted by decreasing volume (largest-first). This is the
        single most reliable greedy warm start for 3D bin packing.

        If allow_rotation=False, restricts each box to its FIRST legal
        orientation (often the natural (v=2, hswap=0)), which is appropriate
        when the dataset disallows reorientation. Vertical-flag constraints
        are ALWAYS honored either way -- this flag only suppresses hswap and
        alternative v values.

        Returns a list of placement dicts. Respects per-type counts."""
        # Build default order: sorted by decreasing volume, replicated by count.
        if box_order is None:
            seq: list[int] = []
            for bt in sorted(type_ids, key=lambda b: -box_value(b)):
                seq.extend([bt] * _count[bt])
            box_order = seq
        else:
            box_order = list(box_order)

        used: dict[int, int] = {bt: 0 for bt in type_ids}
        placements: list[dict] = []
        for bt in box_order:
            if bt not in _count:
                continue
            if used[bt] >= _count[bt]:
                continue
            if allow_rotation:
                orients = None
            else:
                # First legal orientation only (deterministic).
                first = _orient_cache[bt][0]
                orients = [(first[0], first[1])]
            pmt = try_place_at_corner_3d(placements, bt, orientations=orients)
            if pmt is None:
                continue
            placements.append(pmt)
            used[bt] += 1
        return placements

    def wall_building_pack(
        box_order: Optional[Iterable[int]] = None,
        allow_rotation: bool = True,
    ) -> list[dict]:
        """Bischoff-Ratcliff-style WALL BUILDING heuristic. Slices the
        container along its x-axis into 'walls'. Each wall is a thin slab
        of full (W, H) cross-section and adaptive depth (dx). Within a
        wall, boxes are placed by a 2-D shelf packer on the (y, z) face
        with chosen depth = the first box's oriented x-dimension.

        Steps:
          1. Determine box order. Default: largest-volume box first,
             replicated by count.
          2. For each remaining box (in order), open a new wall if needed.
             The wall depth dx is set by the FIRST box placed in it
             (taking the orientation whose y*z fits in (W, H) and whose
             x is the largest x-dim available -- gives the most room).
          3. Inside a wall, run a left-bottom shelf packer on the (y, z)
             face: shelves grow along +z; within a shelf, boxes lay along
             +y. Boxes that don't fit in the current shelf open a new one
             higher up. Boxes whose x-extent exceeds the wall's depth get
             skipped (will be retried in the NEXT wall).
          4. Move to the next wall at x += dx until the container is full
             or all boxes are placed.

        Robust to vertical-flag constraints (orientations are filtered
        through _orient_cache). If allow_rotation=False, only the first
        legal orientation of each box type is used.

        Returns a list of placement dicts (container_id = 0)."""
        if box_order is None:
            seq: list[int] = []
            for bt in sorted(type_ids, key=lambda b: -box_value(b)):
                seq.extend([bt] * _count[bt])
            box_order = seq
        else:
            box_order = list(box_order)
        # remaining[i] = box_type for index i, None if already placed
        remaining: list[Optional[int]] = list(box_order)
        # respect counts: if `box_order` had more of some bt than available,
        # truncate per-type to count
        used: dict[int, int] = {bt: 0 for bt in type_ids}
        for i, bt in enumerate(remaining):
            if bt not in _count:
                remaining[i] = None
                continue
            if used[bt] >= _count[bt]:
                remaining[i] = None
            else:
                used[bt] += 1
        # reset used for actual packing
        used = {bt: 0 for bt in type_ids}

        def _orients_for(bt: int):
            if allow_rotation:
                return _orient_cache[bt]
            return _orient_cache[bt][:1]

        placements: list[dict] = []
        x_cursor = 0
        # safety: bounded number of walls (one per remaining box at worst)
        max_walls = sum(1 for r in remaining if r is not None) + 1
        for _wall_iter in range(max_walls):
            if x_cursor >= L:
                break
            if not any(r is not None for r in remaining):
                break
            # Find the first unplaced box to seed the wall; pick its
            # orientation with the largest x dim that still fits in (W,H).
            seed_idx = None
            for i, bt in enumerate(remaining):
                if bt is None:
                    continue
                if used[bt] >= _count[bt]:
                    remaining[i] = None
                    continue
                # candidate orientations whose (y, z) fit in (W, H) and
                # whose x fits in remaining length
                cands = []
                for (v, hs, sx, sy, sz) in _orients_for(bt):
                    if sy <= W and sz <= H and x_cursor + sx <= L:
                        cands.append((v, hs, sx, sy, sz))
                if not cands:
                    continue
                # pick orientation with the LARGEST sx (deepest wall);
                # ties: smallest sy*sz first to leave room on the face
                cands.sort(key=lambda o: (-o[2], o[3] * o[4]))
                seed_v, seed_hs, seed_sx, seed_sy, seed_sz = cands[0]
                seed_idx = i
                wall_dx = seed_sx
                # place the seed at (x_cursor, 0, 0)
                placements.append({
                    "box_type": int(bt),
                    "container_id": 0,
                    "x": int(x_cursor),
                    "y": 0,
                    "z": 0,
                    "v": int(seed_v),
                    "hswap": int(seed_hs),
                })
                used[bt] += 1
                remaining[i] = None
                break
            if seed_idx is None:
                # nothing fits in any remaining x slack
                break
            # Now build the wall: 2D shelf-pack the (y, z) face within
            # the slab [x_cursor, x_cursor+wall_dx) x [0,W) x [0,H).
            # Shelf state: (z_bottom, shelf_height, y_cursor)
            shelves: list[list[int]] = [[seed_sz, seed_sz, seed_sy]]
            # z_bottom of shelf k is sum of heights of shelves[0..k-1].
            # We track z_bottom explicitly:
            shelf_z = [0]
            shelf_h = [seed_sz]
            shelf_y = [seed_sy]
            # Iterate remaining unplaced boxes for this wall.
            for i, bt in enumerate(remaining):
                if bt is None:
                    continue
                if used[bt] >= _count[bt]:
                    remaining[i] = None
                    continue
                # find an orientation whose x fits in wall_dx
                placed_this = False
                for (v, hs, sx, sy, sz) in _orients_for(bt):
                    if sx > wall_dx:
                        continue
                    # try existing shelves first (best-fit by y leftover)
                    chosen_k = -1
                    chosen_best = None
                    for k in range(len(shelf_z)):
                        if sz <= shelf_h[k] and shelf_y[k] + sy <= W:
                            leftover = W - (shelf_y[k] + sy)
                            if chosen_best is None or leftover < chosen_best:
                                chosen_best = leftover
                                chosen_k = k
                    if chosen_k >= 0:
                        placements.append({
                            "box_type": int(bt),
                            "container_id": 0,
                            "x": int(x_cursor),
                            "y": int(shelf_y[chosen_k]),
                            "z": int(shelf_z[chosen_k]),
                            "v": int(v),
                            "hswap": int(hs),
                        })
                        shelf_y[chosen_k] += sy
                        used[bt] += 1
                        remaining[i] = None
                        placed_this = True
                        break
                    # open a new shelf at the top
                    top_z = shelf_z[-1] + shelf_h[-1] if shelf_z else 0
                    if top_z + sz <= H and sy <= W:
                        shelf_z.append(top_z)
                        shelf_h.append(sz)
                        shelf_y.append(sy)
                        placements.append({
                            "box_type": int(bt),
                            "container_id": 0,
                            "x": int(x_cursor),
                            "y": 0,
                            "z": int(top_z),
                            "v": int(v),
                            "hswap": int(hs),
                        })
                        used[bt] += 1
                        remaining[i] = None
                        placed_this = True
                        break
                if placed_this:
                    continue
                # try next box; leave this one for the next wall (if it
                # could fit deeper) or it stays unplaced forever
            x_cursor += wall_dx
        return placements

    def apply_swap_boxes(
        placements: list[dict],
        time_limit_s: float = 3.0,
        seed: Optional[int] = None,
    ) -> list[dict]:
        """Simple perturb-and-repack local search.

        For up to `time_limit_s` seconds:
          * Take the current `placements`.
          * Randomly remove a small fraction (10-30%) of placed boxes.
          * Re-pack via try_place_at_corner_3d in a shuffled order.
          * Keep the new packing if its used_volume is >= the old one.

        Returns the best placements found (NEW list; input not mutated).
        Useful as a cheap improvement step after wall_building_pack or
        corner_pack_3d. Pure Python; complexity dominated by the
        O(k * placed^2) overlap checks per repack."""
        rng = random.Random(seed)
        best = list(placements)
        best_vol = used_volume(best)
        t0 = time.time()
        safety = 0.05
        while (time.time() - t0) < time_limit_s - safety and best:
            # remove a random subset
            frac = rng.uniform(0.10, 0.30)
            n_rm = max(1, int(len(best) * frac))
            keep_idx = list(range(len(best)))
            rng.shuffle(keep_idx)
            removed_idx = set(keep_idx[:n_rm])
            kept = [best[i] for i in range(len(best)) if i not in removed_idx]
            removed = [best[i] for i in range(len(best)) if i in removed_idx]
            # build a fresh order: first reinsert removed boxes (shuffled),
            # then attempt any UNplaced boxes (those types with leftover
            # count after `kept` is counted).
            used_after_kept: dict[int, int] = {bt: 0 for bt in type_ids}
            for p in kept:
                used_after_kept[p["box_type"]] += 1
            order_bts = [p["box_type"] for p in removed]
            # any unused boxes (volume-largest first)
            for bt in sorted(type_ids, key=lambda b: -box_value(b)):
                leftover = _count[bt] - used_after_kept[bt] - order_bts.count(bt)
                if leftover > 0:
                    order_bts.extend([bt] * leftover)
            rng.shuffle(order_bts)
            new = list(kept)
            for bt in order_bts:
                # respect counts
                if sum(1 for p in new if p["box_type"] == bt) >= _count[bt]:
                    continue
                pmt = try_place_at_corner_3d(new, bt)
                if pmt is not None:
                    new.append(pmt)
            new_vol = used_volume(new)
            if new_vol > best_vol:
                best = new
                best_vol = new_vol
        return best

    # ==================================================================
    # (5) Solution-dict builder + one-shot solver
    # ==================================================================
    def make_solution(placements) -> dict:
        """Wrap a list of placement dicts into the EXACT dict shape eval_func
        expects: {'placements': list[{'box_type','container_id','x','y','z',
        'v','hswap'}, ...]}. Use on the output of wall_building_pack() /
        corner_pack_3d() / apply_swap_boxes() so you never return the wrong
        dict shape."""
        pl = list(placements) if placements else []
        return {"placements": pl}

    def solve_default(time_limit_s: float = 10.0) -> dict:
        """ONE-SHOT STRONG SOLVER. Returns the complete solution dict
        {'placements': [...]} ready to return directly.

        Strategy:
          1. Run both wall_building_pack() and corner_pack_3d() (default
             largest-volume-first order). Keep whichever has higher
             used_volume.
          2. Polish the winner with apply_swap_boxes under the remaining
             time budget (perturb-and-repack hill climber).
          3. Wrap as the eval_func-expected dict.

        Use as the FIRST thing your solve() function calls. ONE LINE:
            return tools['solve_default'](time_limit_s=10)
        """
        import time as _time
        t0 = _time.time()
        # Step 1: two constructive heuristics, pick the better.
        cand_a = wall_building_pack()
        cand_b = corner_pack_3d()
        best = cand_a if used_volume(cand_a) >= used_volume(cand_b) else cand_b
        # Step 2: polish with remaining time.
        remaining = time_limit_s - (_time.time() - t0)
        if remaining > 1.0 and best:
            best = apply_swap_boxes(best, time_limit_s=remaining, seed=0)
        return make_solution(best)

    return {
        # (5) one-shot + builder (CALL FIRST)
        "solve_default": solve_default,
        "make_solution": make_solution,
        # (3) construction
        "wall_building_pack": wall_building_pack,
        "corner_pack_3d": corner_pack_3d,
        "try_place_at_corner_3d": try_place_at_corner_3d,
        # (4) local search
        "apply_swap_boxes": apply_swap_boxes,
        # (2) feasibility primitives
        "overlap_3d": overlap_3d,
        "fits_in_container": fits_in_container,
        "used_volume": used_volume,
        "used_count": used_count,
        "utilization": utilization,
        # (1) queries
        "container_dims": container_dims,
        "box_dims": box_dims,
        "box_value": box_value,
        "box_count_available": box_count_available,
        "n_box_types": n_box_types,
        "box_orientations": box_orientations,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (5) ONE-SHOT STRONG SOLVER (call this first!) -----
    {
        "name": "solve_default",
        "input": "time_limit_s: float = 10.0",
        "output": "dict {'placements': list[placement_dict]}",
        "purpose": (
            "RECOMMENDED START: returns a complete solution dict ready to return "
            "directly. Runs BOTH wall_building_pack and corner_pack_3d, keeps "
            "the higher-volume packing, then polishes with apply_swap_boxes "
            "under the remaining budget. Honors per-type counts and "
            "vertical-flag constraints. ONE LINE: "
            "`return tools['solve_default'](time_limit_s=10)`."
        ),
    },
    {
        "name": "make_solution",
        "input": "placements: Iterable[dict]",
        "output": "dict {'placements': list[dict]}",
        "purpose": (
            "Build the EXACT solution dict shape eval_func wants from a list of "
            "placement dicts. Use on the output of wall_building_pack() / "
            "corner_pack_3d() / apply_swap_boxes() so you never return the wrong "
            "dict shape."
        ),
    },
    # ----- (1) Queries -----
    {
        "name": "container_dims",
        "input": "(no args)",
        "output": "tuple[int, int, int]",
        "purpose": (
            "Container size (L, W, H) along the x / y / z axes. Use to "
            "bound placements: a box at (x,y,z) with oriented size "
            "(sx,sy,sz) must satisfy x+sx<=L, y+sy<=W, z+sz<=H."
        ),
    },
    {
        "name": "box_dims",
        "input": "bt: int  (box type id)",
        "output": "tuple[int, int, int]",
        "purpose": (
            "Raw (unoriented) dimensions (d1, d2, d3) of box type `bt`. "
            "Multiply for volume; pair with box_orientations() to get the "
            "oriented sizes for each legal (v, hswap)."
        ),
    },
    {
        "name": "box_value",
        "input": "bt: int",
        "output": "int",
        "purpose": (
            "Volume d1*d2*d3 of one box of type `bt`. The problem objective "
            "is volume utilization, so this IS the per-box value -- use it "
            "as the sort key for largest-first greedy heuristics."
        ),
    },
    {
        "name": "box_count_available",
        "input": "bt: int",
        "output": "int",
        "purpose": (
            "Number of boxes of type `bt` available to place. CO-Bench "
            "marks the solution infeasible if you exceed this for any "
            "type, so always compare against used_count() before adding."
        ),
    },
    {
        "name": "n_box_types",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of distinct box types in the instance.",
    },
    {
        "name": "box_orientations",
        "input": "bt: int",
        "output": "list[tuple[int, int, int, int, int]]",
        "purpose": (
            "All LEGAL orientations of box type `bt` honoring vertical-flag "
            "constraints. Each tuple is (v, hswap, size_x, size_y, size_z): "
            "the first two are the solution-dict fields, the last three are "
            "the resulting oriented sizes you compare against the container. "
            "Up to 6 entries; fewer when some axes can't be vertical."
        ),
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "overlap_3d",
        "input": "placements: list[dict], candidate: dict",
        "output": "bool",
        "purpose": (
            "True iff `candidate` (a placement dict with box_type/x/y/z/v/"
            "hswap) overlaps any box already in `placements`. Touching "
            "faces are NOT overlap (matches CO-Bench's eval_func). O(k) "
            "where k = len(placements)."
        ),
    },
    {
        "name": "fits_in_container",
        "input": "candidate: dict",
        "output": "bool",
        "purpose": (
            "True iff `candidate` lies fully inside [0,L)x[0,W)x[0,H) and "
            "has a legal (v, hswap). Does NOT check overlap with other "
            "boxes -- pair with overlap_3d for full local feasibility."
        ),
    },
    {
        "name": "used_volume",
        "input": "placements: Iterable[dict]",
        "output": "int",
        "purpose": (
            "Sum of oriented box volumes for the given placements. No "
            "feasibility check; useful as the numerator of utilization()."
        ),
    },
    {
        "name": "used_count",
        "input": "placements: Iterable[dict], box_type: int",
        "output": "int",
        "purpose": (
            "How many boxes of `box_type` are in `placements`. Compare "
            "against box_count_available(bt) before adding another."
        ),
    },
    {
        "name": "utilization",
        "input": "placements: Iterable[dict]",
        "output": "float",
        "purpose": (
            "used_volume / container_volume. Equals CO-Bench's objective "
            "when `placements` is feasible. Use as a fast local proxy for "
            "tools['objective'] inside inner loops."
        ),
    },
    # ----- (3) Construction -----
    {
        "name": "try_place_at_corner_3d",
        "input": ("placements: list[dict], box_type: int, "
                  "orientations: Iterable[(v, hswap)] | None = None"),
        "output": "dict | None",
        "purpose": (
            "Attempt to add ONE box of `box_type` at an extreme-point "
            "anchor (origin + face-front corners of already-placed boxes), "
            "scanning anchors bottom-back-left first and orientations in "
            "the given order. Returns a placement dict on success or None. "
            "Honors vertical-flag constraints. Pass `orientations` to "
            "restrict the search (e.g., disable rotation)."
        ),
    },
    {
        "name": "corner_pack_3d",
        "input": ("box_order: Iterable[int] | None = None, "
                  "allow_rotation: bool = True"),
        "output": "list[dict]",
        "purpose": (
            "Greedy 3D extreme-point packer. Iterates `box_order` (a "
            "sequence of box-type ids, repeats allowed up to per-type "
            "counts) and calls try_place_at_corner_3d for each, skipping "
            "boxes that don't fit. Default order: largest-volume first, "
            "every available box. Set allow_rotation=False to lock each "
            "box into its first legal orientation. Returns a placements "
            "list ready to drop into the solution dict."
        ),
    },
    {
        "name": "wall_building_pack",
        "input": ("box_order: Iterable[int] | None = None, "
                  "allow_rotation: bool = True"),
        "output": "list[dict]",
        "purpose": (
            "Bischoff-Ratcliff WALL-BUILDING heuristic. Splits the "
            "container along x into successive 'walls' (slabs); inside "
            "each wall, runs a 2-D shelf packer on the (y, z) face. The "
            "first box in each wall sets the wall's depth (its oriented "
            "x-size, chosen to be the deepest one that still fits in "
            "(W, H)). Robust to vertical-flag constraints and per-type "
            "counts. Often beats corner_pack_3d when boxes are similar "
            "sized, while corner_pack_3d wins for heterogeneous mixes."
        ),
    },
    # ----- (4) Local search -----
    {
        "name": "apply_swap_boxes",
        "input": ("placements: list[dict], time_limit_s: float = 3.0, "
                  "seed: int | None = None"),
        "output": "list[dict]",
        "purpose": (
            "Perturb-and-repack hill climber. Repeatedly removes a random "
            "10-30%% subset of placed boxes and re-inserts them (plus any "
            "unused boxes) via try_place_at_corner_3d in shuffled order; "
            "keeps the new packing only when it has strictly larger used "
            "volume. Cheap improvement step after wall_building_pack or "
            "corner_pack_3d. Pure Python; budget a few seconds at most "
            "on large instances."
        ),
    },
]
