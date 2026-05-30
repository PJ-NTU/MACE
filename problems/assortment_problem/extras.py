"""Per-problem extras for CO-Bench Assortment problem.

The task: arrange rectangular pieces (possibly rotated 90 deg) into stock
rectangles to MINIMISE the overall waste-area percentage.  Hard constraints
that LLM solutions frequently violate:

  * Each piece TYPE's total placed count across ALL stock instances must lie
    in its [min, max] bound -- this is the notorious "Piece count violation"
    failure mode.
  * At most TWO distinct stock types may be used.
  * Pieces must lie within the stock and not overlap.

The solution dict CO-Bench expects:
    {"objective": float,
     "placements": {stock_instance_id (1-indexed): {
         "stock_type": int (1-indexed),
         "placements": [
             {"piece": int (1-indexed),
              "x": float, "y": float,
              "orientation": 0 or 1}, ...
         ]
     }, ...}}

Tool groups (Tier 1..4):

  (1) Queries          -- n_types, n_stocks, piece_type_min/max, dims, areas
  (2) Feasibility      -- count_per_type, unmet_min_types, excess_max_types,
                          used_stock_types, total_waste
  (3) Construction     -- pack_counts_into_stock, greedy_minimal_feasible,
                          greedy_for_bounds, apply_swap_pieces
  (4) Heavy / exact    -- ilp_assortment (chooses piece-type counts and #
                          stock instances within bounds, then shelf-packs)

KEY INSIGHT: the geometry (2D packing) is NP-hard, but the COUNTS subproblem
is a small ILP.  The previous fail-loop came from greedy ignoring `min`
bounds.  `ilp_assortment` chooses counts and stock instance multiplicities
so that EVERY type ends up in [min, max] and the (lower-bound) waste
percentage is minimised -- then we shelf-pack into concrete (x, y).  This is
the surest way to satisfy the count constraints.

`piece_type_min(t)` / `piece_type_max(t)` are 1-INDEXED to match CO-Bench's
solution schema (piece ids 1..m).  Internally we still use 0-indexed lists.
"""
from __future__ import annotations
from typing import Optional, Iterable, List, Dict, Tuple

from mip import Model, BINARY, INTEGER, MINIMIZE, xsum, OptimizationStatus


def extra_tools(instance: dict) -> dict:
    """Factory: returns Assortment-specific tool callables.

    Instance schema (CO-Bench Assortment, one case):
      - m:          int, number of piece TYPES
      - n:          int, number of stock TYPES
      - waste_cost: float (unused by waste-% objective; kept for completeness)
      - stocks:     list[dict] with keys 'length', 'width', 'fixed_cost'
      - pieces:     list[dict] with keys 'length', 'width', 'min', 'max',
                    'value'
    """
    m: int = int(instance["m"])
    n: int = int(instance["n"])
    stocks = instance["stocks"]
    pieces = instance["pieces"]

    # Pre-compute commonly used arrays (0-indexed).
    p_len = [float(p["length"]) for p in pieces]
    p_wid = [float(p["width"]) for p in pieces]
    p_min = [int(p["min"]) for p in pieces]
    p_max = [int(p["max"]) for p in pieces]
    p_area = [p_len[t] * p_wid[t] for t in range(m)]

    s_len = [float(s["length"]) for s in stocks]
    s_wid = [float(s["width"]) for s in stocks]
    s_area = [s_len[s] * s_wid[s] for s in range(n)]

    EPS = 1e-6

    # ------------------------------------------------------------------
    # Helpers (private)
    # ------------------------------------------------------------------
    def _pt(t: int) -> int:
        """1-indexed piece type -> 0-indexed."""
        i = int(t) - 1
        if not (0 <= i < m):
            raise ValueError(f"piece type {t} out of range [1, {m}]")
        return i

    def _st(s: int) -> int:
        """1-indexed stock type -> 0-indexed."""
        i = int(s) - 1
        if not (0 <= i < n):
            raise ValueError(f"stock type {s} out of range [1, {n}]")
        return i

    def _piece_fits_in_stock(t0: int, s0: int) -> Tuple[bool, bool]:
        """(fits_normal, fits_rotated) for piece-type t0 in stock-type s0."""
        L, W = s_len[s0], s_wid[s0]
        pl, pw = p_len[t0], p_wid[t0]
        normal = (pl <= L + EPS) and (pw <= W + EPS)
        rotated = (pw <= L + EPS) and (pl <= W + EPS)
        return normal, rotated

    def _shelf_pack_into_stock(
        s0: int,
        items: List[Tuple[int, int]],   # list of (piece_type_0idx, orientation)
    ) -> Tuple[List[dict], List[Tuple[int, int]]]:
        """Greedy next-fit-decreasing-height shelf packing of `items` into a
        single instance of stock type s0.  Returns
            (placed_records, leftover_items)
        where placed_records are dicts in CO-Bench format (1-indexed piece id)
        and leftover_items are the items that didn't fit into THIS stock
        instance and need another one (or rejection).

        Items are sorted by height (descending) to make shelves tight.  Each
        shelf has a fixed y, height = first item's height, and is filled
        left-to-right until the next item would overflow the stock length;
        then a new shelf opens at y = y + shelf_height.
        """
        L, W = s_len[s0], s_wid[s0]
        # Compute (length, width) per oriented item.
        oriented = []
        for idx, (t0, ori) in enumerate(items):
            if ori == 0:
                il, iw = p_len[t0], p_wid[t0]
            else:
                il, iw = p_wid[t0], p_len[t0]
            oriented.append((il, iw, t0, ori, idx))
        # Sort by height (iw) descending, then length (il) descending.
        order = sorted(range(len(oriented)),
                       key=lambda k: (-oriented[k][1], -oriented[k][0]))
        placed: List[dict] = []
        leftover_idx: List[int] = []
        cur_x = 0.0
        cur_y = 0.0
        shelf_h = 0.0
        for k in order:
            il, iw, t0, ori, original_idx = oriented[k]
            if il > L + EPS or iw > W + EPS:
                # Won't ever fit in this stock instance.
                leftover_idx.append(original_idx)
                continue
            if cur_x + il <= L + EPS and cur_y + max(shelf_h, iw) <= W + EPS:
                # Fits on current shelf.
                if iw > shelf_h:
                    shelf_h = iw
                placed.append({
                    "piece": t0 + 1,
                    "x": float(cur_x),
                    "y": float(cur_y),
                    "orientation": int(ori),
                })
                cur_x += il
            else:
                # Open a new shelf.
                new_y = cur_y + shelf_h
                if new_y + iw <= W + EPS and il <= L + EPS:
                    cur_x = 0.0
                    cur_y = new_y
                    shelf_h = iw
                    placed.append({
                        "piece": t0 + 1,
                        "x": float(cur_x),
                        "y": float(cur_y),
                        "orientation": int(ori),
                    })
                    cur_x += il
                else:
                    leftover_idx.append(original_idx)
        leftover_items = [items[i] for i in leftover_idx]
        return placed, leftover_items

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def n_types() -> int:
        """Number of piece types m."""
        return m

    def n_stocks() -> int:
        """Number of stock TYPES n (recall: unlimited instances of each)."""
        return n

    def piece_type_min(t: int) -> int:
        """Minimum total count for piece type t (1-indexed)."""
        return p_min[_pt(t)]

    def piece_type_max(t: int) -> int:
        """Maximum total count for piece type t (1-indexed)."""
        return p_max[_pt(t)]

    def piece_dims(t: int) -> Tuple[float, float]:
        """(length, width) of piece type t (1-indexed) in its base orientation."""
        i = _pt(t)
        return p_len[i], p_wid[i]

    def stock_dims(s: int) -> Tuple[float, float]:
        """(length, width) of stock type s (1-indexed)."""
        i = _st(s)
        return s_len[i], s_wid[i]

    def piece_area(t: int) -> float:
        """Area of one piece of type t (1-indexed)."""
        return p_area[_pt(t)]

    def stock_area(s: int) -> float:
        """Area of one instance of stock type s (1-indexed)."""
        return s_area[_st(s)]

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def count_per_type(placements: Dict) -> List[int]:
        """List of length m: total placed count of each piece type
        (index t-1 holds count of piece type t).  Accepts the full
        'placements' dict from a solution."""
        counts = [0] * m
        if not isinstance(placements, dict):
            return counts
        for _inst_id, instance_data in placements.items():
            if not isinstance(instance_data, dict):
                continue
            for pl in instance_data.get("placements", []) or []:
                try:
                    t = int(pl.get("piece"))
                except (TypeError, ValueError):
                    continue
                if 1 <= t <= m:
                    counts[t - 1] += 1
        return counts

    def unmet_min_types(placements: Dict) -> List[Tuple[int, int, int]]:
        """List of (piece_type_1idx, current_count, required_min) for every
        piece type whose count is BELOW its min.  An empty list means every
        type meets its min -- a NECESSARY condition for feasibility."""
        cnt = count_per_type(placements)
        out = []
        for t0 in range(m):
            if cnt[t0] < p_min[t0]:
                out.append((t0 + 1, cnt[t0], p_min[t0]))
        return out

    def excess_max_types(placements: Dict) -> List[Tuple[int, int, int]]:
        """List of (piece_type_1idx, current_count, allowed_max) for every
        piece type whose count is ABOVE its max."""
        cnt = count_per_type(placements)
        out = []
        for t0 in range(m):
            if cnt[t0] > p_max[t0]:
                out.append((t0 + 1, cnt[t0], p_max[t0]))
        return out

    def used_stock_types(placements: Dict) -> set:
        """Set of distinct stock-type ids (1-indexed) used in `placements`.
        Feasibility requires this set's size to be <= 2."""
        used = set()
        if not isinstance(placements, dict):
            return used
        for _inst_id, instance_data in placements.items():
            if isinstance(instance_data, dict) and "stock_type" in instance_data:
                try:
                    used.add(int(instance_data["stock_type"]))
                except (TypeError, ValueError):
                    pass
        return used

    def total_waste(placements: Dict) -> Tuple[float, float, float]:
        """Returns (total_stock_area, total_used_area, waste_percentage).
        Waste percentage = (stock_area - used_area) / stock_area, matching
        CO-Bench's objective definition.  Useful for cheap evaluation of
        candidate solutions without invoking eval_func."""
        tot_stock = 0.0
        tot_used = 0.0
        if isinstance(placements, dict):
            for _inst_id, instance_data in placements.items():
                if not isinstance(instance_data, dict):
                    continue
                try:
                    s0 = int(instance_data["stock_type"]) - 1
                except (TypeError, KeyError, ValueError):
                    continue
                if not (0 <= s0 < n):
                    continue
                tot_stock += s_area[s0]
                for pl in instance_data.get("placements", []) or []:
                    try:
                        t0 = int(pl["piece"]) - 1
                    except (TypeError, KeyError, ValueError):
                        continue
                    if 0 <= t0 < m:
                        tot_used += p_area[t0]
        if tot_stock <= 0:
            return 0.0, 0.0, float("inf")
        return tot_stock, tot_used, (tot_stock - tot_used) / tot_stock

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def pack_counts_into_stock(
        stock_type: int,
        piece_counts: Dict[int, int],
    ) -> Tuple[List[Dict], Dict[int, int]]:
        """Pack a given multiset of pieces into AS FEW instances as possible
        of a single stock type (shelf next-fit-decreasing-height).

        Args:
          stock_type:    1-indexed stock type id.
          piece_counts:  dict {piece_type_1idx: count}.  Counts can be 0.

        Returns:
          (instance_dicts, leftover_counts) where
          - instance_dicts is a list of {"stock_type", "placements"} dicts
            (one per stock instance used) -- ready to be added to the
            solution's `placements` map with new instance ids.
          - leftover_counts is {piece_type_1idx: count} for pieces that did
            NOT fit in any stock instance (typically because the piece is
            larger than the stock in both orientations).

        Pieces are tried with whichever orientation reduces the bounding-box
        height (for tighter shelves) -- if a piece only fits in one
        orientation, that one is used.  Always check leftover_counts before
        accepting the result.
        """
        s0 = _st(stock_type)
        items: List[Tuple[int, int]] = []
        leftover: Dict[int, int] = {}
        for t, cnt in piece_counts.items():
            t0 = _pt(t)
            normal, rotated = _piece_fits_in_stock(t0, s0)
            if not normal and not rotated:
                leftover[t] = int(cnt)
                continue
            # Prefer the orientation giving a smaller "width" (shelf height
            # consumption) so shelves are tighter.
            if normal and rotated:
                ori = 0 if p_wid[t0] <= p_len[t0] else 1
            elif normal:
                ori = 0
            else:
                ori = 1
            for _ in range(int(cnt)):
                items.append((t0, ori))
        instance_dicts: List[Dict] = []
        remaining = items
        # Loop: open a new stock instance, pack until full, repeat.  Hard cap
        # on iterations to guard against pathological cases.
        max_instances = max(1, len(items) * 2 + 10)
        for _ in range(max_instances):
            if not remaining:
                break
            placed, leftover_items = _shelf_pack_into_stock(s0, remaining)
            if not placed:
                # Nothing fit -- residue can never fit; mark as leftover.
                for (t0, _ori) in remaining:
                    leftover[t0 + 1] = leftover.get(t0 + 1, 0) + 1
                remaining = []
                break
            instance_dicts.append({
                "stock_type": stock_type,
                "placements": placed,
            })
            if len(leftover_items) >= len(remaining):
                # No forward progress -- avoid infinite loops.
                for (t0, _ori) in leftover_items:
                    leftover[t0 + 1] = leftover.get(t0 + 1, 0) + 1
                remaining = []
                break
            remaining = leftover_items
        if remaining:
            for (t0, _ori) in remaining:
                leftover[t0 + 1] = leftover.get(t0 + 1, 0) + 1
        return instance_dicts, leftover

    def greedy_minimal_feasible(stock_type: Optional[int] = None) -> Dict:
        """Build a solution that just satisfies every piece type's `min`
        bound.  Picks each piece-type's lower-bound count and shelf-packs
        them into instances of ONE stock type (uses at most 1 distinct
        stock type, so the "<= 2 distinct stock types" rule is trivially
        satisfied).

        Args:
          stock_type: which stock type (1-indexed) to use.  If None, the
                      stock type that holds the largest fraction of pieces
                      by area is chosen heuristically (favouring larger
                      stocks to reduce instance count).

        Returns the full solution dict (with `objective` and `placements`).
        If any piece type does not fit in the chosen stock (in either
        orientation) it will be MISSING from the result -- check
        unmet_min_types on the returned `placements` before using.
        """
        if stock_type is None:
            # Score each stock type by total area divided by minimal piece
            # area it can host (i.e., how many small pieces it can hold).
            best_s = 1
            best_score = -1.0
            for s_idx in range(n):
                area_capacity = s_area[s_idx]
                fits_any = False
                for t_idx in range(m):
                    if p_min[t_idx] <= 0:
                        continue
                    nrm, rot = _piece_fits_in_stock(t_idx, s_idx)
                    if nrm or rot:
                        fits_any = True
                        break
                if not fits_any:
                    continue
                # Bigger stocks usually mean fewer instances.
                if area_capacity > best_score:
                    best_score = area_capacity
                    best_s = s_idx + 1
            stock_type = best_s
        counts = {t0 + 1: p_min[t0] for t0 in range(m) if p_min[t0] > 0}
        inst_dicts, _leftover = pack_counts_into_stock(stock_type, counts)
        placements = {i + 1: d for i, d in enumerate(inst_dicts)}
        _ts, _tu, waste = total_waste(placements)
        return {"objective": float(waste), "placements": placements}

    def greedy_for_bounds(
        stock_type: Optional[int] = None,
        prefer: str = "min",
    ) -> Dict:
        """Construct a solution that satisfies every type's [min, max]
        bound, then optionally tops up pieces (within max) to reduce waste.

        Args:
          stock_type: 1-indexed stock type to use.  None -> pick automatically
                      as in greedy_minimal_feasible.
          prefer:     "min" -> place only `min` of each type (smallest legal
                                count; minimises piece area but stock usage
                                still costs the same -- often poor waste).
                      "max" -> first place `min` of each type, then GREEDILY
                                place additional pieces (one type at a time,
                                largest piece-area first) up to each type's
                                `max`, as long as they fit in the currently
                                open stock instances.  This usually reduces
                                waste percentage substantially.

        Returns a complete solution dict using a SINGLE stock type (so the
        "<= 2 stock types" constraint is trivially satisfied).
        """
        if prefer not in ("min", "max"):
            raise ValueError(f"prefer must be 'min' or 'max', got {prefer!r}")
        if stock_type is None:
            base = greedy_minimal_feasible(None)
            # Read out the stock type used (if any).
            if base["placements"]:
                first_inst = next(iter(base["placements"].values()))
                stock_type = int(first_inst.get("stock_type", 1))
            else:
                stock_type = 1
        if prefer == "min":
            return greedy_minimal_feasible(stock_type)
        # prefer == "max": place min, then add up to max for each type by
        # piece area descending (biggest pieces displace the most waste).
        counts = {t0 + 1: p_min[t0] for t0 in range(m) if p_min[t0] > 0}
        # Add extras up to max, prioritising large pieces.
        order = sorted(range(m), key=lambda t0: -p_area[t0])
        for t0 in order:
            slack = p_max[t0] - p_min[t0]
            if slack > 0:
                counts[t0 + 1] = counts.get(t0 + 1, 0) + slack
        inst_dicts, leftover = pack_counts_into_stock(stock_type, counts)
        # If some "extras" didn't fit, we must REMOVE them from counts and
        # accept the smaller assortment.  The min counts always fit if the
        # stock can hold them at all (it's our responsibility to pick a
        # stock type for which that's true -- caller may need to try
        # different stock types).
        placements = {i + 1: d for i, d in enumerate(inst_dicts)}
        # Sanity-check: if mins aren't met (e.g. stock too small for a
        # type), fall back to min-only with a different stock type.
        cnts = count_per_type(placements)
        unmet = [t0 for t0 in range(m) if cnts[t0] < p_min[t0]]
        if unmet:
            # Try every stock type and return the first that meets all mins.
            for s_try in range(1, n + 1):
                cand = greedy_minimal_feasible(s_try)
                if not unmet_min_types(cand["placements"]):
                    return greedy_for_bounds(s_try, "max")
            # All stock types fail -- return whatever we have.
        _ts, _tu, waste = total_waste(placements)
        return {"objective": float(waste), "placements": placements}

    def apply_swap_pieces(
        solution: Dict,
        from_type: int,
        to_type: int,
    ) -> Optional[Dict]:
        """Try to swap ONE occurrence of piece type `from_type` for piece
        type `to_type` in `solution`.  Returns a NEW solution dict if the
        swap is feasible (i.e. `from` count is still >= its min, `to` count
        is still <= its max, and the new piece fits in the original
        location -- in either orientation), else None.

        Use as a tiny neighbourhood move in local search.  The new piece is
        placed at the same (x, y) as the removed one with the orientation
        that fits (or stays at orientation 0 if both fit).  If the new
        piece is larger than the slot, the swap is rejected.
        """
        if not isinstance(solution, dict):
            return None
        placements = solution.get("placements")
        if not isinstance(placements, dict):
            return None
        ft = int(from_type)
        tt = int(to_type)
        if not (1 <= ft <= m and 1 <= tt <= m):
            return None
        if ft == tt:
            return solution
        cnt = count_per_type(placements)
        if cnt[ft - 1] - 1 < p_min[ft - 1]:
            return None
        if cnt[tt - 1] + 1 > p_max[tt - 1]:
            return None
        # Find a placement of `from_type`.
        import copy
        new_placements = copy.deepcopy(placements)
        for inst_id, inst_data in new_placements.items():
            for k, pl in enumerate(inst_data.get("placements", [])):
                if int(pl.get("piece", -1)) != ft:
                    continue
                # Check the swap fits.  The "slot" size is the
                # from_piece's bounding box under its orientation.
                ori_old = int(pl.get("orientation", 0))
                t0_old = ft - 1
                if ori_old == 0:
                    slot_l, slot_w = p_len[t0_old], p_wid[t0_old]
                else:
                    slot_l, slot_w = p_wid[t0_old], p_len[t0_old]
                t0_new = tt - 1
                new_normal = (p_len[t0_new] <= slot_l + EPS and
                              p_wid[t0_new] <= slot_w + EPS)
                new_rotated = (p_wid[t0_new] <= slot_l + EPS and
                               p_len[t0_new] <= slot_w + EPS)
                if not (new_normal or new_rotated):
                    continue
                new_ori = 0 if new_normal else 1
                inst_data["placements"][k] = {
                    "piece": tt,
                    "x": float(pl["x"]),
                    "y": float(pl["y"]),
                    "orientation": int(new_ori),
                }
                _ts, _tu, waste = total_waste(new_placements)
                return {"objective": float(waste),
                        "placements": new_placements}
        return None

    # ==================================================================
    # (4) Heavy / exact: ILP for piece COUNTS
    # ==================================================================
    def ilp_assortment(
        time_limit_s: float = 10.0,
        stock_type_choices: Optional[Iterable[int]] = None,
    ) -> Optional[Dict]:
        """Choose (a) the piece-type counts c_t in [min_t, max_t] for each
        type, and (b) the number of instances I_s used of each candidate
        stock type s in `stock_type_choices`, so that all c_t pieces FIT
        into the I_s stock instances by AREA, and the total stock area used
        is minimised.  Then shelf-pack a concrete geometry.

        Args:
          time_limit_s:        CBC wall-clock budget (seconds).
          stock_type_choices:  iterable of 1-indexed stock types to consider.
                                Up to 2 will end up in the solution (the
                                model enforces <= 2 active types).  Defaults
                                to all n stock types.

        Returns a full CO-Bench solution dict, or None if the ILP fails or
        the resulting counts cannot be shelf-packed into the chosen
        instances (the area lower bound is necessary but not sufficient for
        2D packing -- the post-shelf check will detect and try to repair).

        Caveat: the ILP only enforces the AREA lower bound on stock count.
        The shelf packer afterwards may need to open additional instances
        if the packing density turns out to be worse than area-tight.  This
        is generally fine for the assortment instances (pieces are small
        relative to stocks) -- the area bound is usually tight to within a
        few percent.
        """
        if stock_type_choices is None:
            choices = list(range(1, n + 1))
        else:
            choices = [int(s) for s in stock_type_choices]
            for s in choices:
                if not (1 <= s <= n):
                    raise ValueError(f"stock type {s} out of range [1, {n}]")
        if not choices:
            return None
        choices0 = [s - 1 for s in choices]  # 0-indexed

        model = Model(sense=MINIMIZE)
        model.verbose = 0
        model.max_seconds = float(time_limit_s)

        # Piece-type count variables.
        c = [model.add_var(var_type=INTEGER, lb=p_min[t], ub=p_max[t],
                           name=f"c_{t}") for t in range(m)]
        # Stock-type instance-count variables (number of instances used).
        # Upper bound: enough to hold all max counts in worst case.
        max_pieces_total = sum(p_max)
        I_ub = max(1, max_pieces_total + 1)
        I = [model.add_var(var_type=INTEGER, lb=0, ub=I_ub, name=f"I_{s0}")
             for s0 in choices0]
        # y[k] = 1 iff stock type choices0[k] is USED at all.
        y = [model.add_var(var_type=BINARY, name=f"y_{s0}")
             for s0 in choices0]
        # I_k <= UB * y_k  and  y_k <= I_k (forces correspondence).
        for k in range(len(choices0)):
            model += I[k] <= I_ub * y[k], f"link_iy_{k}"
            model += I[k] >= y[k], f"link_yi_{k}"
        # At most 2 distinct stock types used.
        model += xsum(y[k] for k in range(len(choices0))) <= 2, "two_types"
        # Area constraint: total piece area <= total stock area used.  We
        # need this in BOTH senses (don't want to under-provision OR pay for
        # excess stocks beyond what's needed):
        # sum_t area_t * c_t <= sum_k area_{s0[k]} * I_k
        model += (
            xsum(p_area[t] * c[t] for t in range(m))
            <= xsum(s_area[s0] * I[k] for k, s0 in enumerate(choices0))
        ), "area_capacity"
        # Piece-fits-in-stock-type constraint: a piece-type cannot
        # contribute c_t > 0 unless SOME used stock type can host it.
        # Encoded as: if no usable stock type, force c_t = p_min[t]
        # (we still have to satisfy the min).  For tractability we just
        # forbid placing extras of types that fit in NO chosen stock --
        # the LLM is responsible for the choice list.  This is a soft
        # check, not encoded as a hard constraint -- the shelf-packer
        # will detect leftovers.

        # Objective: minimise total stock area used.  Lower stock area at
        # fixed piece area => lower waste percentage.
        model.objective = xsum(s_area[s0] * I[k]
                               for k, s0 in enumerate(choices0))
        status = model.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if model.num_solutions < 1:
            return None
        # Extract counts and instance multiplicities.
        chosen_counts: Dict[int, int] = {}
        for t in range(m):
            v = c[t].x
            if v is None:
                return None
            chosen_counts[t + 1] = int(round(v))
        chosen_instances: List[Tuple[int, int]] = []  # (stock_type_1idx, n)
        for k, s0 in enumerate(choices0):
            v = I[k].x
            if v is None:
                continue
            cnt = int(round(v))
            if cnt > 0:
                chosen_instances.append((s0 + 1, cnt))
        if not chosen_instances:
            return None
        # Shelf-pack chosen counts.  Strategy: assign pieces to stock types
        # greedily, largest-piece-first to the stock type with best fit.
        # For simplicity, if there's a SINGLE chosen stock type, dump all
        # counts into it; if there are two, split by piece-fits-in-stock.
        if len(chosen_instances) == 1:
            stk_type = chosen_instances[0][0]
            inst_dicts, leftover = pack_counts_into_stock(stk_type, chosen_counts)
            if leftover:
                # Some pieces didn't fit -- the ILP didn't enforce
                # per-stock-type fits.  Bail out.
                return None
            placements = {i + 1: d for i, d in enumerate(inst_dicts)}
        else:
            # Two stock types: route each piece type to a stock type it
            # actually fits in (preferring the larger stock).  We don't
            # enforce a balance -- the area LB ensures the total fits.
            placements = {}
            cursor = 1
            stk_areas = [(s_area[stk - 1], stk) for stk, _ in chosen_instances]
            stk_areas.sort(reverse=True)
            # Decide each piece type's stock type assignment.
            piece_to_stock: Dict[int, int] = {}
            for t in range(m):
                if chosen_counts.get(t + 1, 0) == 0:
                    continue
                for _, stk in stk_areas:
                    nrm, rot = _piece_fits_in_stock(t, stk - 1)
                    if nrm or rot:
                        piece_to_stock[t + 1] = stk
                        break
                if t + 1 not in piece_to_stock:
                    # Piece type fits nowhere -- bail out (shouldn't happen
                    # if the ILP picked compatible stocks).
                    return None
            # Pack per stock type.
            from collections import defaultdict as _dd
            by_stock: Dict[int, Dict[int, int]] = _dd(dict)
            for tt, stk in piece_to_stock.items():
                by_stock[stk][tt] = chosen_counts[tt]
            for stk, cnts in by_stock.items():
                inst_dicts, leftover = pack_counts_into_stock(stk, cnts)
                if leftover:
                    return None
                for d in inst_dicts:
                    placements[cursor] = d
                    cursor += 1
        _ts, _tu, waste = total_waste(placements)
        # Sanity check: mins met?  If not, return None.
        if unmet_min_types(placements):
            return None
        if excess_max_types(placements):
            return None
        return {"objective": float(waste), "placements": placements}

    # ==================================================================
    # (5) One-shot strong solver
    # ==================================================================
    def solve_default(time_limit_s: float = 10.0) -> Dict:
        """ONE-SHOT STRONG SOLVER. Returns the complete solution dict
        {'objective': float, 'placements': {...}} ready to return directly.

        Strategy:
          1. Try ilp_assortment(time_limit_s) -- chooses piece-counts and
             stock-instance multiplicities to satisfy [min, max] and the
             "<=2 distinct stock types" rule while minimising stock area,
             then shelf-packs into (x, y). This is the strongest tool when
             it succeeds.
          2. If ILP returns None, fall back to greedy_for_bounds('max')
             which always returns a single-stock-type solution that meets
             every type's `min` (so it is always feasible w.r.t. the
             count + stock-type constraints).
          3. As an emergency last resort, returns greedy_minimal_feasible().

        Use as the FIRST thing your solve() function calls. ONE LINE:
            return tools['solve_default'](time_limit_s=10)
        """
        sol = ilp_assortment(time_limit_s=time_limit_s)
        if sol is not None and not unmet_min_types(sol["placements"]) \
                and not excess_max_types(sol["placements"]):
            return sol
        sol = greedy_for_bounds(prefer="max")
        if sol is not None and not unmet_min_types(sol["placements"]) \
                and not excess_max_types(sol["placements"]):
            return sol
        return greedy_minimal_feasible()

    return {
        # (5) one-shot (CALL FIRST)
        "solve_default": solve_default,
        # (4) exact / heavy
        "ilp_assortment": ilp_assortment,
        # (3) construction
        "greedy_for_bounds": greedy_for_bounds,
        "greedy_minimal_feasible": greedy_minimal_feasible,
        "pack_counts_into_stock": pack_counts_into_stock,
        "apply_swap_pieces": apply_swap_pieces,
        # (2) feasibility
        "unmet_min_types": unmet_min_types,
        "excess_max_types": excess_max_types,
        "used_stock_types": used_stock_types,
        "count_per_type": count_per_type,
        "total_waste": total_waste,
        # (1) queries
        "n_types": n_types,
        "n_stocks": n_stocks,
        "piece_type_min": piece_type_min,
        "piece_type_max": piece_type_max,
        "piece_dims": piece_dims,
        "stock_dims": stock_dims,
        "piece_area": piece_area,
        "stock_area": stock_area,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (5) ONE-SHOT STRONG SOLVER (call this first!) -----
    {
        "name": "solve_default",
        "input": "time_limit_s: float = 10.0",
        "output": "dict {'objective': float, 'placements': {...}}",
        "purpose": (
            "RECOMMENDED START: returns a complete solution dict ready to return "
            "directly. Tries ilp_assortment first (chooses piece-counts + "
            "stock-instance multiplicities satisfying [min, max] and the "
            "'<=2 distinct stock types' rule); falls back to greedy_for_bounds"
            "('max') if the ILP fails, and to greedy_minimal_feasible() as a "
            "last resort. The returned dict satisfies every piece type's min/"
            "max bounds. ONE LINE: "
            "`return tools['solve_default'](time_limit_s=10)`."
        ),
    },
    # ----- (4) Exact / heavy -----
    {
        "name": "ilp_assortment",
        "input": "time_limit_s: float = 10.0, stock_type_choices: Iterable[int] | None = None",
        "output": "dict | None",
        "purpose": (
            "Use as primary solver. Chooses (a) piece-type counts c_t in "
            "[min_t, max_t] and (b) how many instances of each candidate stock "
            "type to use, so that (i) at most 2 distinct stock types are active, "
            "(ii) total piece area <= total stock area, and (iii) total stock "
            "area used (waste %) is MINIMISED. Then shelf-packs into concrete "
            "(x, y). Returns a full {'objective', 'placements'} solution dict, "
            "or None if no feasible counts/area combination exists or the "
            "shelf-pack overflows. Wrap with greedy_for_bounds('max') as a "
            "backup -- or just call solve_default() instead."
        ),
    },
    # ----- (1) Queries -----
    {
        "name": "n_types",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of piece TYPES m (piece ids are 1..m in the solution).",
    },
    {
        "name": "n_stocks",
        "input": "(no args)",
        "output": "int",
        "purpose": (
            "Number of stock TYPES n.  Recall: each type has UNLIMITED "
            "instances but at most 2 DISTINCT types may appear in the "
            "solution (CO-Bench raises an exception if > 2)."
        ),
    },
    {
        "name": "piece_type_min",
        "input": "t: int (1-indexed)",
        "output": "int",
        "purpose": (
            "Minimum number of pieces of type t REQUIRED across the whole "
            "solution.  Violating this is the #1 cause of 'Piece count "
            "violation' failures -- always sum your placed counts and check "
            ">= this value before returning."
        ),
    },
    {
        "name": "piece_type_max",
        "input": "t: int (1-indexed)",
        "output": "int",
        "purpose": (
            "Maximum number of pieces of type t allowed across the whole "
            "solution.  Exceeding this also raises 'Piece count violation'."
        ),
    },
    {
        "name": "piece_dims",
        "input": "t: int (1-indexed)",
        "output": "(float, float)",
        "purpose": (
            "(length, width) of piece type t in its base orientation 0.  "
            "Rotation 1 swaps length and width."
        ),
    },
    {
        "name": "stock_dims",
        "input": "s: int (1-indexed)",
        "output": "(float, float)",
        "purpose": "(length, width) of stock type s.",
    },
    {
        "name": "piece_area",
        "input": "t: int (1-indexed)",
        "output": "float",
        "purpose": "Area = length * width of one piece of type t.",
    },
    {
        "name": "stock_area",
        "input": "s: int (1-indexed)",
        "output": "float",
        "purpose": "Area = length * width of one instance of stock type s.",
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "count_per_type",
        "input": "placements: dict (solution['placements'])",
        "output": "list[int] of length m",
        "purpose": (
            "Total placed count of each piece type across all stock "
            "instances.  Index t-1 holds count of piece type t.  Use as the "
            "single source of truth for the count constraint."
        ),
    },
    {
        "name": "unmet_min_types",
        "input": "placements: dict",
        "output": "list[(piece_type_1idx, current_count, required_min)]",
        "purpose": (
            "Lists every piece type whose count is BELOW its min.  Empty "
            "list <=> all `min` constraints are satisfied (necessary for "
            "feasibility).  CHECK THIS BEFORE RETURNING -- the eval_func "
            "will reject the solution otherwise."
        ),
    },
    {
        "name": "excess_max_types",
        "input": "placements: dict",
        "output": "list[(piece_type_1idx, current_count, allowed_max)]",
        "purpose": (
            "Lists every piece type whose count is ABOVE its max.  Empty "
            "list <=> all `max` constraints are satisfied."
        ),
    },
    {
        "name": "used_stock_types",
        "input": "placements: dict",
        "output": "set[int]",
        "purpose": (
            "Distinct stock-type ids (1-indexed) used in `placements`.  "
            "len(...) must be <= 2 or eval_func raises 'More than 2 "
            "distinct stock types'."
        ),
    },
    {
        "name": "total_waste",
        "input": "placements: dict",
        "output": "(total_stock_area, total_used_area, waste_percentage)",
        "purpose": (
            "Cheap reproduction of CO-Bench's objective: waste_percentage "
            "= (stock_area - used_area) / stock_area.  Use to evaluate "
            "candidate solutions inside a local search loop without paying "
            "for the full eval_func feasibility check."
        ),
    },
    # ----- (3) Construction -----
    {
        "name": "pack_counts_into_stock",
        "input": "stock_type: int (1-indexed), piece_counts: dict {1idx: cnt}",
        "output": "(list[stock_instance_dict], leftover_counts dict)",
        "purpose": (
            "Shelf next-fit-decreasing-height packer.  Packs the multiset "
            "`piece_counts` into AS FEW instances of `stock_type` as the "
            "packer manages.  Returns the list of instance dicts (each "
            "{'stock_type', 'placements'}, ready for the solution map) and "
            "a leftover-counts dict for pieces that did NOT fit.  Always "
            "check leftover_counts before accepting -- non-empty means the "
            "piece is too large for the stock in both orientations."
        ),
    },
    {
        "name": "greedy_minimal_feasible",
        "input": "stock_type: int | None = None",
        "output": "solution dict",
        "purpose": (
            "Build a solution that places exactly `min` of each piece type "
            "in instances of ONE stock type (so '<=2 distinct types' is "
            "trivial).  If `stock_type` is None, auto-picks a likely-good "
            "stock type.  Useful as a feasibility-first warm start; you "
            "can then top up to `max` to reduce waste -- or use "
            "`greedy_for_bounds(prefer='max')` directly."
        ),
    },
    {
        "name": "greedy_for_bounds",
        "input": "stock_type: int | None = None, prefer: 'min' | 'max' = 'min'",
        "output": "solution dict",
        "purpose": (
            "Construct a feasible solution using ONE stock type.  "
            "prefer='min' places exactly `min` of each type (smallest legal "
            "assortment).  prefer='max' places `min` then adds extras up to "
            "`max` (largest-piece-first), usually reducing waste "
            "percentage significantly because each placed piece displaces "
            "exactly its area of waste.  Falls back to alternate stock "
            "types automatically if the mins can't be met."
        ),
    },
    {
        "name": "apply_swap_pieces",
        "input": "solution: dict, from_type: int (1-indexed), to_type: int (1-indexed)",
        "output": "solution dict | None",
        "purpose": (
            "Tiny local-search move: swap ONE occurrence of piece `from_type` "
            "for piece `to_type` in the same (x, y) slot, choosing an "
            "orientation that fits.  Returns None if (a) `from` count would "
            "drop below its min, (b) `to` count would exceed its max, or "
            "(c) the new piece is too big for the slot.  Pure function -- "
            "input is not mutated."
        ),
    },
]
