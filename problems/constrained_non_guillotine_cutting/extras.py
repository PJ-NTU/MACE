"""Per-problem extras for CO-Bench Constrained Non-Guillotine Cutting.

The task is 2-D rectangular packing on a single stock sheet without the
guillotine (single-straight-cut) restriction. Each piece type has integer
demand bounds [min, max] and a value; the goal is to maximize total value
of placed pieces. Pieces may be rotated 90 degrees.

Tools fall in 4 tiers (mirroring the bin_packing_one_dimensional / TSP /
set_partitioning extras layout):

  (1) Queries:        stock_dims, piece_dims, piece_value,
                      piece_demand_min, piece_demand_max, n_piece_types

  (2) Feasibility
      primitives:     overlap_with_placed, used_count_per_type,
                      total_area_used, candidate_corners,
                      is_feasible_solution

  (3) Construction /
      improvement:    bottom_left_pack_BFD, greedy_max_value_density,
                      try_place_at_corner, apply_swap_pieces

  (4) Heavy:          (omitted -- 2-D NGC ILP is too large to be useful
                      as a tool. Use Tier 3 constructors and local moves.)

CO-Bench solution schema:
    {"placements": [(piece_type, x, y, r), ...]}
  - piece_type is 1-BASED.
  - (x, y) is the bottom-left corner of the placed rectangle.
  - r in {0, 1}: 0 = no rotation (use length x width), 1 = rotated 90 deg
    (use width x length).

All tools accept / return placements in this exact form so results plug
straight into the solution dict.
"""
from __future__ import annotations

import random
import time
from typing import Iterable, Optional


def extra_tools(instance: dict) -> dict:
    """Factory: returns CNGC-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench load_data):
      - stock_length: int   (sheet width along x)
      - stock_width:  int   (sheet height along y)
      - pieces:       list[dict]  each with keys:
            'length', 'width', 'min', 'max', 'value'
    """
    L = int(instance["stock_length"])
    W = int(instance["stock_width"])
    pieces = instance["pieces"]
    T = len(pieces)  # number of piece types

    # Precompute per-type arrays so tier-1 queries are O(1).
    p_len = [int(p["length"]) for p in pieces]
    p_wid = [int(p["width"]) for p in pieces]
    p_val = [int(p["value"]) for p in pieces]
    p_min = [int(p["min"]) for p in pieces]
    p_max = [int(p["max"]) for p in pieces]
    p_area = [p_len[t] * p_wid[t] for t in range(T)]

    # ==================================================================
    # Helpers (closure-private)
    # ==================================================================
    def _check_type(t: int):
        if not (1 <= int(t) <= T):
            raise ValueError(f"piece_type {t} out of range [1, {T}]")

    def _dims(t: int, r: int) -> tuple[int, int]:
        """Effective (length-along-x, width-along-y) for type t with rot r.
        Type is 1-based to match the solution schema."""
        _check_type(t)
        if r not in (0, 1):
            raise ValueError(f"rotation flag {r} must be 0 or 1")
        if r == 0:
            return p_len[t - 1], p_wid[t - 1]
        return p_wid[t - 1], p_len[t - 1]

    def _placement_rect(placement) -> tuple[int, int, int, int, int]:
        """Decode a placement tuple to (type, x1, y1, x2, y2)."""
        t, x, y, r = placement
        dl, dw = _dims(int(t), int(r))
        return int(t), int(x), int(y), int(x) + dl, int(y) + dw

    def _rects_of(placements) -> list[tuple[int, int, int, int]]:
        return [(_placement_rect(p)[1], _placement_rect(p)[2],
                 _placement_rect(p)[3], _placement_rect(p)[4])
                for p in placements]

    def _rect_overlaps(a, b) -> bool:
        """a, b are (x1, y1, x2, y2) half-open rectangles."""
        return not (a[2] <= b[0] or b[2] <= a[0]
                    or a[3] <= b[1] or b[3] <= a[1])

    def _candidate_in_bounds(x: int, y: int, dl: int, dw: int) -> bool:
        return x >= 0 and y >= 0 and (x + dl) <= L and (y + dw) <= W

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def stock_dims() -> tuple[int, int]:
        """Stock sheet dimensions (stock_length, stock_width). Pieces must
        live in the half-open rectangle [0, L) x [0, W) -- equivalently a
        placement's right/top edge must be <= L / W."""
        return (L, W)

    def piece_dims(t: int) -> tuple[int, int]:
        """Native (length, width) of piece type `t` (1-based, NO rotation).
        If rotated (r=1), swap the two values when placing."""
        _check_type(t)
        return (p_len[t - 1], p_wid[t - 1])

    def piece_value(t: int) -> int:
        """Per-unit value contributed by one placement of piece type `t`."""
        _check_type(t)
        return p_val[t - 1]

    def piece_demand_min(t: int) -> int:
        """Minimum number of pieces of type `t` that MUST be placed.
        Below this the solution is infeasible (eval_func raises)."""
        _check_type(t)
        return p_min[t - 1]

    def piece_demand_max(t: int) -> int:
        """Maximum number of pieces of type `t` allowed. Above this the
        solution is infeasible."""
        _check_type(t)
        return p_max[t - 1]

    def n_piece_types() -> int:
        """Number of distinct piece types T (1-based ids run 1..T)."""
        return T

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def overlap_with_placed(placements: Iterable, candidate_rect) -> bool:
        """Does `candidate_rect` overlap any rectangle in `placements`?

        candidate_rect may be either a 4-tuple (x1, y1, x2, y2) already in
        absolute coordinates, OR a (piece_type, x, y, r) placement tuple --
        the helper auto-detects by length 4 vs values.
        Returns True if there is any positive-area intersection."""
        # detect format
        cand = tuple(candidate_rect)
        if len(cand) == 4 and all(isinstance(v, int) for v in cand) \
                and cand[2] >= cand[0] and cand[3] >= cand[1]:
            # Could still be a placement; disambiguate by checking that
            # piece_type would be in [1, T] and r in {0,1}. If both fit, we
            # trust the absolute-rect interpretation when the resulting
            # rect makes geometric sense AND interpreting as a placement
            # would not. To keep behavior predictable, prefer the "rect"
            # interpretation when x2 > x1 and y2 > y1 AND values look like
            # rect coords (x2 may exceed L for stale data, etc.).
            # Disambiguation rule: if cand[3] in {0, 1} and 1 <= cand[0] <= T
            # we treat it as a placement; otherwise as a rect.
            if cand[3] in (0, 1) and 1 <= cand[0] <= T \
                    and not (cand[2] > cand[0] and cand[3] >= cand[1]
                             and cand[2] <= L and cand[3] <= W):
                _, x1, y1, x2, y2 = _placement_rect(cand)
                rect = (x1, y1, x2, y2)
            else:
                rect = cand
        else:
            _, x1, y1, x2, y2 = _placement_rect(cand)
            rect = (x1, y1, x2, y2)
        for p in placements:
            _, ax1, ay1, ax2, ay2 = _placement_rect(p)
            if _rect_overlaps(rect, (ax1, ay1, ax2, ay2)):
                return True
        return False

    def used_count_per_type(placements: Iterable) -> list[int]:
        """Counts[t-1] = number of placements of piece type t in the list.
        Returns a length-T list (0-indexed) so you can compare directly to
        piece_demand_min/max."""
        counts = [0] * T
        for p in placements:
            t = int(p[0])
            if 1 <= t <= T:
                counts[t - 1] += 1
        return counts

    def total_area_used(placements: Iterable) -> int:
        """Sum of placed-piece areas (no overlap check). Useful as a quick
        proxy for 'how full is the sheet': stock area = L*W is the cap."""
        s = 0
        for p in placements:
            t, x, y, r = p
            dl, dw = _dims(int(t), int(r))
            s += dl * dw
        return s

    def candidate_corners(placements: Iterable) -> list[tuple[int, int]]:
        """Bottom-left-staircase candidate points where a new piece could be
        anchored. Returns the set {(0, 0)} U {right-edge corners} U
        {top-edge corners} of every already-placed rectangle, clipped to
        the stock and deduplicated. These are the natural anchor points
        for a bottom-left placement heuristic on the NGC problem."""
        pts: set[tuple[int, int]] = {(0, 0)}
        for p in placements:
            _, x1, y1, x2, y2 = _placement_rect(p)
            for cand in ((x2, y1), (x1, y2), (x2, y2), (0, y2), (x2, 0)):
                cx, cy = cand
                if 0 <= cx < L and 0 <= cy < W:
                    pts.add((int(cx), int(cy)))
        # also corners of placed rects' interior tops touching bottom of
        # other rects help reach all "staircase" notches in dense packs;
        # the above (right, top, both, axis-projection) set is usually
        # sufficient for BL heuristics.
        return sorted(pts, key=lambda q: (q[1], q[0]))

    def is_feasible_solution(solution: dict) -> tuple[bool, Optional[str]]:
        """Local feasibility check that mirrors CO-Bench's eval_func without
        the framework round-trip. Returns (True, None) or (False, reason).
        Faster than tools['is_feasible'] inside tight neighborhood loops."""
        if not isinstance(solution, dict):
            return False, f"solution must be dict, got {type(solution).__name__}"
        placements = solution.get("placements")
        if not isinstance(placements, list):
            return False, "solution['placements'] must be a list"
        counts = [0] * T
        rects: list[tuple[int, int, int, int]] = []
        for idx, pl in enumerate(placements):
            if not (isinstance(pl, (list, tuple)) and len(pl) == 4):
                return False, f"placement {idx} not a 4-tuple"
            t, x, y, r = pl
            if not all(isinstance(v, int) for v in (t, x, y, r)):
                return False, f"placement {idx} has non-int values"
            if not (1 <= t <= T):
                return False, f"placement {idx} invalid piece_type {t}"
            if r not in (0, 1):
                return False, f"placement {idx} invalid rotation {r}"
            dl, dw = _dims(t, r)
            if x < 0 or y < 0 or (x + dl) > L or (y + dw) > W:
                return False, f"placement {idx} out of stock bounds"
            rects.append((x, y, x + dl, y + dw))
            counts[t - 1] += 1
        n = len(rects)
        for i in range(n):
            for j in range(i + 1, n):
                if _rect_overlaps(rects[i], rects[j]):
                    return False, f"placements {i} and {j} overlap"
        for t in range(T):
            if counts[t] < p_min[t] or counts[t] > p_max[t]:
                return False, (f"piece type {t+1} count {counts[t]} "
                               f"outside [{p_min[t]}, {p_max[t]}]")
        return True, None

    # ==================================================================
    # (3) Construction / improvement
    # ==================================================================
    def try_place_at_corner(placements: list, piece_type: int,
                            allow_rotation: bool = True) -> Optional[tuple]:
        """Try to place ONE piece of `piece_type` at the lowest-leftmost
        free corner of the current partial packing.

        Tries (in order): each candidate corner from `candidate_corners`,
        scanned in (y, x) ascending order, with rotation r=0 then r=1 (if
        rotation is allowed and the rotated shape differs). Returns the
        placement tuple (t, x, y, r) for the first feasible position
        found, or None if the piece cannot be placed anywhere.

        This is the workhorse for any constructive heuristic: feed it
        pieces in some priority order and accumulate the returned
        placements. Caller is responsible for respecting per-type max
        demand."""
        _check_type(piece_type)
        rotations = [0, 1] if allow_rotation else [0]
        # avoid trying the same effective shape twice for squares
        seen_shapes: set[tuple[int, int]] = set()
        corners = candidate_corners(placements)
        for cx, cy in corners:
            for r in rotations:
                dl, dw = _dims(piece_type, r)
                if (dl, dw) in seen_shapes and r == 1:
                    continue
                if not _candidate_in_bounds(cx, cy, dl, dw):
                    continue
                rect = (cx, cy, cx + dl, cy + dw)
                bad = False
                for q in placements:
                    _, ax1, ay1, ax2, ay2 = _placement_rect(q)
                    if _rect_overlaps(rect, (ax1, ay1, ax2, ay2)):
                        bad = True
                        break
                if not bad:
                    return (int(piece_type), int(cx), int(cy), int(r))
                seen_shapes.add((dl, dw))
        return None

    def bottom_left_pack_BFD(piece_order: Optional[Iterable[int]] = None,
                             allow_rotation: bool = True) -> list:
        """Bottom-Left First-Fit-Decreasing for 2-D NGC.

        Repeatedly pick types from `piece_order` (default: all types sorted
        by area descending, each repeated up to its piece_demand_max),
        and try to place at the lowest-leftmost free corner (using
        try_place_at_corner). A type is dropped from the queue once it
        cannot be placed anywhere OR its max demand is exhausted.

        Returns a list of placement 4-tuples (NOT a full solution dict --
        wrap as {'placements': result} when returning). Note: the result
        may VIOLATE per-type min-demand if the sheet is too small; callers
        should check used_count_per_type vs piece_demand_min and either
        repair or fall back. For the common ngcutcon instances the area-
        descending order combined with BL placement usually satisfies
        all mins, but no guarantee."""
        if piece_order is None:
            queue: list[int] = sorted(range(1, T + 1),
                                      key=lambda t: -p_area[t - 1])
        else:
            queue = [int(t) for t in piece_order]
        placements: list[tuple[int, int, int, int]] = []
        counts = [0] * T
        # Build a usage budget per type
        budget = [p_max[t - 1] for t in range(1, T + 1)]
        # Iterate the queue, attempting each type once per pass; if any
        # type was placed this pass, do another pass. This lets us pack
        # many copies of a single small high-value type once big ones
        # run out of space.
        active = [t for t in queue if budget[t - 1] > 0]
        # de-dup the queue but preserve relative priority order
        seen: set[int] = set()
        unique_order: list[int] = []
        for t in active:
            if t not in seen:
                seen.add(t)
                unique_order.append(t)
        # Greedy: for each priority type, place as many as possible at the
        # lowest-leftmost free corner before moving on.
        for t in unique_order:
            while counts[t - 1] < p_max[t - 1]:
                pl = try_place_at_corner(placements, t,
                                         allow_rotation=allow_rotation)
                if pl is None:
                    break
                placements.append(pl)
                counts[t - 1] += 1
        return placements

    def greedy_max_value_density(allow_rotation: bool = True) -> list:
        """Same as bottom_left_pack_BFD but with priority order = piece
        value / area descending (the classic 2-D knapsack heuristic).
        Returns a list of placement tuples."""
        # avoid division by zero on degenerate pieces
        order = sorted(range(1, T + 1),
                       key=lambda t: -(p_val[t - 1] / max(p_area[t - 1], 1)))
        return bottom_left_pack_BFD(piece_order=order,
                                    allow_rotation=allow_rotation)

    def apply_swap_pieces(placements: list, time_limit_s: float = 2.0,
                          rng_seed: Optional[int] = None) -> list:
        """Single-piece improvement move: try to REPLACE each placed piece
        with a higher-value piece type (any type, any rotation) that fits
        in the same bounding rectangle of the removed piece OR at the same
        bottom-left corner. Iterates until no improving swap exists or
        time_limit_s elapses.

        Operates on a value-greedy basis (first improvement). Returns a
        NEW list of placements (never mutates the input). Respects per-
        type max demand. NOTE: it does NOT introduce new placements, only
        swaps existing ones -- so total count of placed pieces is invariant.
        For inserting brand-new pieces, call try_place_at_corner on the
        returned list."""
        rng = random.Random(rng_seed)
        t0 = time.time()
        cur = [tuple(int(v) for v in pl) for pl in placements]
        improved = True
        while improved and (time.time() - t0) < time_limit_s - 0.01:
            improved = False
            order = list(range(len(cur)))
            rng.shuffle(order)
            for i in order:
                if (time.time() - t0) >= time_limit_s - 0.01:
                    break
                old_t, ox, oy, _ = cur[i]
                # remove i, recompute counts
                counts = [0] * T
                for j, pl in enumerate(cur):
                    if j == i:
                        continue
                    counts[int(pl[0]) - 1] += 1
                rest = [pl for j, pl in enumerate(cur) if j != i]
                best_gain = 0
                best_new = None
                for new_t in range(1, T + 1):
                    if counts[new_t - 1] >= p_max[new_t - 1]:
                        continue
                    gain = p_val[new_t - 1] - p_val[old_t - 1]
                    if gain <= best_gain:
                        continue
                    for r in (0, 1):
                        dl, dw = _dims(new_t, r)
                        # try placing at the same anchor (ox, oy)
                        if not _candidate_in_bounds(ox, oy, dl, dw):
                            continue
                        rect = (ox, oy, ox + dl, oy + dw)
                        bad = False
                        for q in rest:
                            _, ax1, ay1, ax2, ay2 = _placement_rect(q)
                            if _rect_overlaps(rect, (ax1, ay1, ax2, ay2)):
                                bad = True
                                break
                        if not bad:
                            best_gain = gain
                            best_new = (new_t, ox, oy, r)
                            break
                if best_new is not None:
                    # check min-demand isn't broken (would removing the
                    # old type drop us below its min?)
                    new_counts = list(counts)
                    new_counts[best_new[0] - 1] += 1
                    if new_counts[old_t - 1] >= p_min[old_t - 1]:
                        cur = rest + [best_new]
                        improved = True
                        break
        return [tuple(pl) for pl in cur]

    return {
        # (1) queries
        "stock_dims": stock_dims,
        "piece_dims": piece_dims,
        "piece_value": piece_value,
        "piece_demand_min": piece_demand_min,
        "piece_demand_max": piece_demand_max,
        "n_piece_types": n_piece_types,
        # (2) feasibility primitives
        "overlap_with_placed": overlap_with_placed,
        "used_count_per_type": used_count_per_type,
        "total_area_used": total_area_used,
        "candidate_corners": candidate_corners,
        "is_feasible_solution": is_feasible_solution,
        # (3) construction / improvement
        "try_place_at_corner": try_place_at_corner,
        "bottom_left_pack_BFD": bottom_left_pack_BFD,
        "greedy_max_value_density": greedy_max_value_density,
        "apply_swap_pieces": apply_swap_pieces,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "stock_dims",
        "input": "(no args)",
        "output": "tuple[int, int]  (stock_length, stock_width)",
        "purpose": (
            "Dimensions of the rectangular stock sheet. A placement (t, x, "
            "y, r) is in-bounds iff x>=0, y>=0, x+dl<=L, y+dw<=W where "
            "(dl, dw) = piece_dims after applying rotation r."
        ),
    },
    {
        "name": "piece_dims",
        "input": "t: int  (1-based piece type)",
        "output": "tuple[int, int]  (length, width) -- NO rotation applied",
        "purpose": (
            "Native dimensions of piece type `t`. To get the effective "
            "footprint of a rotated placement (r=1), swap the two values."
        ),
    },
    {
        "name": "piece_value",
        "input": "t: int",
        "output": "int",
        "purpose": (
            "Value contributed by ONE placement of piece type `t`. "
            "Objective = sum of values over all placements."
        ),
    },
    {
        "name": "piece_demand_min",
        "input": "t: int",
        "output": "int",
        "purpose": (
            "Minimum number of pieces of type `t` that MUST appear in any "
            "feasible solution. Often 0 in the ngcutcon dataset, but not "
            "always -- always check before pruning a type from your queue."
        ),
    },
    {
        "name": "piece_demand_max",
        "input": "t: int",
        "output": "int",
        "purpose": (
            "Maximum number of pieces of type `t` allowed. Exceeding this "
            "makes the solution infeasible."
        ),
    },
    {
        "name": "n_piece_types",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of distinct piece types T (1-based ids run 1..T).",
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "overlap_with_placed",
        "input": "placements: list[placement], candidate_rect: tuple",
        "output": "bool",
        "purpose": (
            "True iff `candidate_rect` overlaps ANY placement in the list. "
            "`candidate_rect` may be either a placement 4-tuple (t, x, y, r) "
            "or an absolute rectangle (x1, y1, x2, y2). Use this when you "
            "consider an anchor point and want a fast 'does this fit?' "
            "check without recomputing the full feasibility."
        ),
    },
    {
        "name": "used_count_per_type",
        "input": "placements: list[placement]",
        "output": "list[int]  (length T, 0-indexed)",
        "purpose": (
            "counts[t-1] = how many times piece type `t` appears in the "
            "list. Compare against piece_demand_min/max to detect quota "
            "violations BEFORE returning the solution."
        ),
    },
    {
        "name": "total_area_used",
        "input": "placements: list[placement]",
        "output": "int",
        "purpose": (
            "Sum of placed-piece areas. Upper bound is stock_length * "
            "stock_width. Used as a quick density proxy in pruning."
        ),
    },
    {
        "name": "candidate_corners",
        "input": "placements: list[placement]",
        "output": "list[tuple[int, int]]  sorted by (y, x) ascending",
        "purpose": (
            "Bottom-left-staircase anchor candidates for a new piece: "
            "(0, 0) plus every (x2, y1), (x1, y2), (x2, y2), and axis-"
            "projected corner from existing placements, clipped to the "
            "stock. These are the natural points to test for any BL "
            "placement heuristic on a non-guillotine sheet."
        ),
    },
    {
        "name": "is_feasible_solution",
        "input": "solution: dict",
        "output": "(bool, str | None)",
        "purpose": (
            "Local feasibility check (bounds + no-overlap + per-type "
            "count in [min, max]). Mirrors CO-Bench's eval_func semantics "
            "without the framework round-trip; faster than tools["
            "'is_feasible'] for tight neighborhood-search loops."
        ),
    },
    # ----- (3) Construction / improvement -----
    {
        "name": "try_place_at_corner",
        "input": ("placements: list[placement], piece_type: int, "
                  "allow_rotation: bool = True"),
        "output": "placement tuple (t, x, y, r) | None",
        "purpose": (
            "Try to place ONE piece of `piece_type` at the lowest-leftmost "
            "free corner of the current partial packing (uses "
            "candidate_corners + overlap check). Returns the placement "
            "tuple of the FIRST feasible position found, or None if no "
            "corner accepts the piece in either rotation. Workhorse for "
            "any constructive heuristic: call repeatedly while respecting "
            "per-type max demand."
        ),
    },
    {
        "name": "bottom_left_pack_BFD",
        "input": ("piece_order: Iterable[int] | None = None  "
                  "(default: all types by area desc), "
                  "allow_rotation: bool = True"),
        "output": "list[placement]  (NOT a solution dict -- wrap before return)",
        "purpose": (
            "Bottom-Left First-Fit-Decreasing for 2-D NGC: iterates "
            "`piece_order` and for each type, calls try_place_at_corner "
            "repeatedly until either the type's max-demand is met or no "
            "anchor accepts it. Returns a list of placements. WARNING: "
            "the result may violate per-type MIN-demand if the sheet is "
            "too small -- always check used_count_per_type against "
            "piece_demand_min and either repair or fall back to a "
            "different order. Wrap as {'placements': result} before "
            "returning from solve()."
        ),
    },
    {
        "name": "greedy_max_value_density",
        "input": "allow_rotation: bool = True",
        "output": "list[placement]",
        "purpose": (
            "bottom_left_pack_BFD with priority order = value / area "
            "descending (classic 2-D knapsack heuristic). Often beats "
            "area-desc on ngcutcon when value/area varies widely."
        ),
    },
    {
        "name": "apply_swap_pieces",
        "input": ("placements: list[placement], time_limit_s: float = 2.0, "
                  "rng_seed: int | None = None"),
        "output": "list[placement]  (new list; input not mutated)",
        "purpose": (
            "Single-piece IMPROVEMENT move: try to replace each placed "
            "piece with a higher-value type/rotation that fits at the "
            "same bottom-left anchor and respects per-type max demand. "
            "First-improvement; iterates until no swap exists or "
            "time_limit_s elapses. Keeps the total number of placements "
            "constant -- to add new pieces afterwards, call "
            "try_place_at_corner on the returned list."
        ),
    },
]
