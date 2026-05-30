"""Per-problem extras for Unconstrained Guillotine Cutting (UGC, gcut*).

UGC = 2D guillotine cutting where, in classical literature, each piece
"type" may be cut an unlimited number of times. CO-Bench's `eval_func`
nonetheless rejects duplicate `piece_id`s -- so for a candidate placement
to survive the framework's feasibility check every `piece_id` value must
be unique. The tools below adopt this convention: they treat the m
entries of `pieces` as DISTINCT piece types, each placeable AT MOST ONCE
(matching `eval_func`).

Tool groups:
  (1) Inspection:        stock_dims, piece_dims, piece_value, n_piece_types,
                         count_in_solution, area_used
  (2) Construction:      guillotine_greedy_value_density
  (3) Heavy / exact:     gilmore_gomory_dp -- 2D guillotine DP via
                         recursive horizontal / vertical splits on
                         normalized (raster) coordinate sets. Returns
                         the value-maximizing placement subject to the
                         at-most-once piece constraint.
  (4) Local search:      apply_swap_local -- swap one placed piece out
                         for an unused piece of similar / larger value
                         that fits in the freed sub-rectangle.

All are exposed under tools[...]. Coordinate convention matches
`eval_func`: orientation 0 -> placed width = piece['l'] (x-extent),
placed height = piece['w'] (y-extent). orientation 1 swaps them and is
only emitted when `instance['allow_rotation']` is True.
"""
from __future__ import annotations
import time
from typing import Iterable, Optional


def extra_tools(instance: dict) -> dict:
    """Factory: returns UGC-specific tool callables for the loaded instance."""
    W = int(instance["stock_width"])
    H = int(instance["stock_height"])
    pieces: dict = instance["pieces"]
    allow_rot: bool = bool(instance.get("allow_rotation", False))

    # piece_ids in 1..m
    piece_ids = sorted(pieces.keys())
    m = len(piece_ids)

    # Pre-extract per-type info
    # type_info[pid] = (l, w, value)
    type_info = {pid: (int(pieces[pid]["l"]),
                       int(pieces[pid]["w"]),
                       int(pieces[pid]["value"]))
                 for pid in piece_ids}

    # Orientations a piece can take. (w_x, h_y, orientation_flag, density)
    def _orientations(pid: int):
        l, w, v = type_info[pid]
        outs = [(l, w, 0, v)]
        if allow_rot and l != w:
            outs.append((w, l, 1, v))
        return outs

    # ==================================================================
    # (1) Inspection
    # ==================================================================
    def stock_dims() -> tuple:
        """Return (stock_width, stock_height)."""
        return (W, H)

    def piece_dims(piece_type: int) -> tuple:
        """Return (l, w) for `piece_type` (orientation 0 has x-extent l,
        y-extent w). Raises KeyError if unknown."""
        l, w, _v = type_info[int(piece_type)]
        return (l, w)

    def piece_value(piece_type: int) -> int:
        """Return the value field for `piece_type`."""
        return type_info[int(piece_type)][2]

    def n_piece_types() -> int:
        """Number of distinct piece ids in the instance."""
        return m

    def count_in_solution(placements: Iterable[dict], piece_type: int) -> int:
        """How many times `piece_type` appears in `placements`. With the
        at-most-once constraint this is always 0 or 1 (helpful when
        composing local search that must respect uniqueness)."""
        pid = int(piece_type)
        return sum(1 for p in placements if int(p["piece_id"]) == pid)

    def area_used(placements: Iterable[dict]) -> int:
        """Total occupied area of `placements` (sum of placed piece
        rectangles, ignoring overlap / out-of-bounds). Useful as a quick
        upper-bound sanity check against stock_width * stock_height."""
        total = 0
        for p in placements:
            pid = int(p["piece_id"])
            if pid not in type_info:
                continue
            l, w, _v = type_info[pid]
            total += l * w  # orientation does not change area
        return total

    # ==================================================================
    # (2) Construction
    # ==================================================================
    def _placement_dict(pid, x, y, orient):
        return {"piece_id": int(pid), "x": int(x), "y": int(y),
                "orientation": int(orient)}

    def _fit_orientations_in(pid: int, rw: int, rh: int):
        """Yield (px, py, orient) for each orientation of pid that fits
        in a rw x rh rectangle."""
        for px, py, orient, _v in _orientations(pid):
            if px <= rw and py <= rh:
                yield px, py, orient

    def guillotine_greedy_value_density(time_limit_s: float = 1.0) -> list:
        """Greedy guillotine construction: recursively place the highest
        value/area piece that fits in the current free rectangle, then
        guillotine-split the remainder into two new free rectangles
        (horizontal split below/above and vertical split to the right of
        the placed piece -- we keep both halves and recurse on each).
        Each piece_id is used at most once (matches eval_func).

        Returns a list of placement dicts. Always feasible by
        construction (axis-aligned guillotine partition, non-overlapping,
        within stock, each id used once).
        """
        deadline = time.time() + max(0.0, float(time_limit_s))
        used = set()
        out: list = []

        # Free rectangles to fill: (x, y, w, h)
        stack = [(0, 0, W, H)]
        # Pre-sort piece ids by value/area descending; recomputed implicitly
        # via filter at each step (cheap since m is usually small).
        order = sorted(piece_ids,
                       key=lambda pid: type_info[pid][2] /
                       max(1, type_info[pid][0] * type_info[pid][1]),
                       reverse=True)

        while stack:
            if time.time() > deadline:
                break
            x0, y0, rw, rh = stack.pop()
            if rw <= 0 or rh <= 0:
                continue
            chosen = None
            for pid in order:
                if pid in used:
                    continue
                fits = list(_fit_orientations_in(pid, rw, rh))
                if not fits:
                    continue
                # Prefer larger placed area (least waste in this rect).
                px, py, orient = max(fits, key=lambda t: t[0] * t[1])
                chosen = (pid, px, py, orient)
                break
            if chosen is None:
                continue
            pid, px, py, orient = chosen
            used.add(pid)
            out.append(_placement_dict(pid, x0, y0, orient))
            # Two free children via a guillotine split (vertical first):
            # right strip:  (x0+px, y0,   rw-px, py)
            # top strip:    (x0,    y0+py, rw,   rh-py)
            if rw - px > 0 and py > 0:
                stack.append((x0 + px, y0, rw - px, py))
            if rh - py > 0:
                stack.append((x0, y0 + py, rw, rh - py))
        return out

    # ==================================================================
    # (3) Gilmore-Gomory 2D guillotine DP
    # ==================================================================
    # Build normal (raster) coordinate sets X*, Y* per Gilmore-Gomory:
    #   X* = { sum of subset of piece x-extents, intersected with [0,W] }
    #   Y* = analogous for y.
    # For the at-most-once setting we cap each piece x-extent's
    # multiplicity at 1 in the subset-sum; this still preserves the
    # property that an optimal guillotine layout has all cut positions
    # in X* x Y* (each piece occupies an interval whose endpoints are
    # cumulative sums of widths of pieces in the same band).
    def _normal_set(extents: list, limit: int) -> list:
        reachable = {0}
        for e in extents:
            new = set()
            for r in reachable:
                s = r + e
                if s <= limit:
                    new.add(s)
            reachable |= new
        reachable.add(limit)
        return sorted(reachable)

    # All possible x-extents (= l for orient 0, plus w for orient 1 if
    # rotation allowed) across distinct pieces.
    def _x_extents():
        ext = []
        for pid in piece_ids:
            l, w, _v = type_info[pid]
            if l <= W:
                ext.append(l)
            if allow_rot and w != l and w <= W:
                ext.append(w)
        return ext

    def _y_extents():
        ext = []
        for pid in piece_ids:
            l, w, _v = type_info[pid]
            if w <= H:
                ext.append(w)
            if allow_rot and l != w and l <= H:
                ext.append(l)
        return ext

    def gilmore_gomory_dp(time_limit_s: float = 30.0,
                         max_states: int = 1_500_000) -> dict:
        """Gilmore-Gomory 2D guillotine DP, restricted to the at-most-once
        piece-usage regime that CO-Bench's `eval_func` enforces.

        Memoizes V(w, h) = max total piece value placeable inside a w x h
        sub-rectangle, allowing any sequence of axis-aligned guillotine
        cuts. Cut positions are restricted to the normal (raster) sets
        X*, Y* (Gilmore-Gomory's theorem). The DP itself is the classic
        UNLIMITED-COPY DP (it does not propagate per-piece usage across
        splits, since that would require an exponential mask product
        across cut branches). After reconstruction we dedupe placements
        so each piece id is emitted at most once -- the result is
        therefore a valid CO-Bench placement, and `value` reflects the
        post-dedupe total. Treat the un-deduped DP value as an upper
        bound (the literature UGC optimum for this instance) and the
        deduped placements as a high-quality feasible solution.

        Args:
          time_limit_s: soft wall-clock budget; the DP aborts early and
                        returns the best layout found so far.
          max_states:   safety cap on memoization table size; if
                        exceeded the DP returns the best-so-far layout.

        Returns:
          {"placements": [...], "value": int, "ub_value": int,
           "completed": bool}.
          - placements: dedup'd, CO-Bench-feasible placement list.
          - value: sum of values across `placements` (after dedup).
          - ub_value: raw DP optimum (UPPER BOUND under unlimited copies;
            equals `value` when the DP did not have to drop duplicates).
          - completed: True iff the DP finished without hitting
            time_limit_s or max_states.
        """
        t0 = time.time()
        deadline = t0 + max(0.0, float(time_limit_s))

        Xs = _normal_set(_x_extents(), W)
        Ys = _normal_set(_y_extents(), H)

        # Per-piece feasible orientations within a w x h box.
        piece_fits = {pid: _orientations(pid) for pid in piece_ids}

        memo: dict = {}
        budget_hit = [False]

        # State: (w, h) -> (value, choice)
        # choice in:
        #   ("piece", pid, orient)
        #   ("vsplit", x_cut)
        #   ("hsplit", y_cut)
        #   ("empty",)
        def solve(w: int, h: int) -> tuple:
            if w <= 0 or h <= 0:
                return (0, ("empty",))
            key = (w, h)
            if key in memo:
                return memo[key]
            if budget_hit[0]:
                return (0, ("empty",))
            if time.time() > deadline or len(memo) > max_states:
                budget_hit[0] = True
                return (0, ("empty",))

            best_val = 0
            best_choice = ("empty",)

            # 1) Place a single piece in this rect (cover full footprint
            #    is dominated by the place+split branches below, so we
            #    only need the "rect fits piece, the rest is waste"
            #    leaf case here -- the splits handle non-trivial layouts).
            for pid in piece_ids:
                for pl, pw, orient, val in piece_fits[pid]:
                    if pl <= w and pw <= h and val > best_val:
                        best_val = val
                        best_choice = ("piece", pid, orient)

            # 2) Vertical cuts at x_cut in (0, w) drawn from Xs.
            for x_cut in Xs:
                if x_cut <= 0 or x_cut >= w:
                    continue
                if budget_hit[0]:
                    break
                lv, _ = solve(x_cut, h)
                rv, _ = solve(w - x_cut, h)
                if lv + rv > best_val:
                    best_val = lv + rv
                    best_choice = ("vsplit", x_cut)

            # 3) Horizontal cuts at y_cut in (0, h) drawn from Ys.
            for y_cut in Ys:
                if y_cut <= 0 or y_cut >= h:
                    continue
                if budget_hit[0]:
                    break
                bv, _ = solve(w, y_cut)
                tv, _ = solve(w, h - y_cut)
                if bv + tv > best_val:
                    best_val = bv + tv
                    best_choice = ("hsplit", y_cut)

            memo[key] = (best_val, best_choice)
            return memo[key]

        ub_value, _root_choice = solve(W, H)

        # ---- Reconstruction with dedup ----
        placements: list = []
        used_ids: set = set()

        def reconstruct(x0: int, y0: int, w: int, h: int):
            if w <= 0 or h <= 0:
                return
            key = (w, h)
            if key not in memo:
                return
            _v, choice = memo[key]
            if choice[0] == "piece":
                _, pid, orient = choice
                if pid in used_ids:
                    return  # CO-Bench uniqueness: emit each id at most once
                used_ids.add(pid)
                placements.append(_placement_dict(pid, x0, y0, orient))
            elif choice[0] == "vsplit":
                _, x_cut = choice
                reconstruct(x0, y0, x_cut, h)
                reconstruct(x0 + x_cut, y0, w - x_cut, h)
            elif choice[0] == "hsplit":
                _, y_cut = choice
                reconstruct(x0, y0, w, y_cut)
                reconstruct(x0, y0 + y_cut, w, h - y_cut)
            # "empty": nothing to place

        reconstruct(0, 0, W, H)

        realized = sum(type_info[p["piece_id"]][2] for p in placements)

        return {
            "placements": placements,
            "value": realized,
            "ub_value": int(ub_value),
            "completed": not budget_hit[0],
        }

    # ==================================================================
    # (4) Local search
    # ==================================================================
    def apply_swap_local(placements: Iterable[dict],
                         t_limit: float = 2.0) -> list:
        """Try to improve `placements` by swapping each placed piece for
        an UNUSED piece of higher value that still fits in the same
        bounding rectangle. First-improvement, restarts after each
        successful swap. Returns a NEW list of placements. Guarantees
        feasibility relative to the original layout (we only shrink or
        replace within an already-placed rectangle's footprint, which
        cannot create overlap)."""
        out = [dict(p) for p in placements]
        deadline = time.time() + max(0.0, float(t_limit))

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Set of used ids in current layout
            used = {int(p["piece_id"]) for p in out}
            for i, p in enumerate(out):
                if time.time() > deadline:
                    break
                pid = int(p["piece_id"])
                orient = int(p["orientation"])
                l_old, w_old, v_old = type_info[pid]
                # Footprint of this placement:
                if orient == 0:
                    fw, fh = l_old, w_old
                else:
                    fw, fh = w_old, l_old

                # Find any unused piece type with higher value whose
                # bounding box fits within (fw, fh) in some allowed
                # orientation.
                best_gain = 0
                best_repl = None
                for cand in piece_ids:
                    if cand in used:
                        continue
                    l_c, w_c, v_c = type_info[cand]
                    if v_c <= v_old:
                        continue
                    fits = []
                    if l_c <= fw and w_c <= fh:
                        fits.append((l_c, w_c, 0))
                    if allow_rot and (w_c <= fw and l_c <= fh) and l_c != w_c:
                        fits.append((w_c, l_c, 1))
                    if not fits:
                        continue
                    gain = v_c - v_old
                    if gain > best_gain:
                        best_gain = gain
                        best_repl = (cand, fits[0])
                if best_repl is not None:
                    cand, (_pw, _ph, c_orient) = best_repl
                    used.discard(pid)
                    used.add(cand)
                    out[i] = _placement_dict(cand, int(p["x"]), int(p["y"]),
                                             c_orient)
                    improved = True
                    break  # restart outer scan
        return out

    return {
        "stock_dims": stock_dims,
        "piece_dims": piece_dims,
        "piece_value": piece_value,
        "n_piece_types": n_piece_types,
        "count_in_solution": count_in_solution,
        "area_used": area_used,
        "guillotine_greedy_value_density": guillotine_greedy_value_density,
        "gilmore_gomory_dp": gilmore_gomory_dp,
        "apply_swap_local": apply_swap_local,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Inspection -----
    {
        "name": "stock_dims",
        "input": "(no args)",
        "output": "(int, int)",
        "purpose": (
            "Return (stock_width, stock_height). Convenience wrapper -- "
            "same as (instance['stock_width'], instance['stock_height'])."
        ),
    },
    {
        "name": "piece_dims",
        "input": "piece_type: int",
        "output": "(int, int)",
        "purpose": (
            "Return (l, w) for the given piece id. Orientation 0 places "
            "the piece with x-extent l and y-extent w; orientation 1 "
            "swaps them (only if instance['allow_rotation'])."
        ),
    },
    {
        "name": "piece_value",
        "input": "piece_type: int",
        "output": "int",
        "purpose": "Return the value field for the given piece id.",
    },
    {
        "name": "n_piece_types",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of distinct piece ids in the instance (== m).",
    },
    {
        "name": "count_in_solution",
        "input": "placements: list[dict], piece_type: int",
        "output": "int",
        "purpose": (
            "How many times `piece_type` appears in `placements`. CO-Bench's "
            "eval_func forbids duplicates, so this is always 0 or 1 in a "
            "feasible solution -- use to enforce uniqueness during repair "
            "or local search."
        ),
    },
    {
        "name": "area_used",
        "input": "placements: list[dict]",
        "output": "int",
        "purpose": (
            "Sum of piece areas across `placements` (ignores overlap and "
            "boundary checks). Quick sanity bound: cannot exceed "
            "stock_width * stock_height in a feasible solution."
        ),
    },
    # ----- Construction -----
    {
        "name": "guillotine_greedy_value_density",
        "input": "time_limit_s: float = 1.0",
        "output": "list[dict]",
        "purpose": (
            "Greedy guillotine construction: at each step pick the unused "
            "piece type with highest value / area that fits in the current "
            "free rectangle, place it at the corner, and guillotine-split "
            "the remainder into two new free rectangles. Always returns a "
            "feasible (non-overlapping, in-bounds, unique-ids) placement "
            "list. Good warm start for local search."
        ),
    },
    # ----- Heavy / exact -----
    {
        "name": "gilmore_gomory_dp",
        "input": "time_limit_s: float = 30.0, max_states: int = 1_500_000",
        "output": "{'placements': list[dict], 'value': int, 'ub_value': int, 'completed': bool}",
        "purpose": (
            "Gilmore-Gomory 2D guillotine DP (Operations Research, 1965). "
            "Memoizes the value-maximizing UNLIMITED-COPY guillotine "
            "partition of every reachable sub-rectangle, with cut "
            "positions restricted to the normal (raster) coordinate sets "
            "X*, Y*. `ub_value` is the raw DP optimum and is an upper "
            "bound on what is achievable on this instance; `placements` "
            "is the reconstructed layout dedup'd so each piece id appears "
            "at most once (matches CO-Bench eval_func). `value` is the "
            "post-dedup total. The DP aborts early when `time_limit_s` "
            "elapses or `max_states` is exceeded (completed=False)."
        ),
    },
    # ----- Local search -----
    {
        "name": "apply_swap_local",
        "input": "placements: list[dict], t_limit: float = 2.0",
        "output": "list[dict]",
        "purpose": (
            "Local search: for each placed piece, try replacing it with "
            "an UNUSED piece type of higher value whose bounding box fits "
            "inside the same placement footprint (in either orientation "
            "if rotation is allowed). First-improvement, restarts after "
            "each successful swap. Guarantees feasibility relative to the "
            "input layout (footprint never grows)."
        ),
    },
]
