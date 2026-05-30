"""Per-problem extras for CO-Bench Constrained Guillotine Cutting.

Provides building-block tools so the LLM can compose construction + repair
heuristics for 2-D guillotine cutting (Cluster C: 2-D packing) without
re-deriving low-level geometry, demand bookkeeping, and the guillotine
feasibility recursion from scratch.

CO-Bench problem (Fixed-orientation Constrained Guillotine Cutting):
  * Rectangular stock of size (stock_length x stock_width).
  * m piece types; each has (length, width, max, value).
  * Place axis-aligned, non-overlapping pieces inside the stock; the layout
    must admit a recursive guillotine partition (every sub-region either is
    empty, equals a single placed piece, or has at least one full-span
    edge-to-edge cut that does not slice any rectangle).
  * Each piece type t may appear at most piece_demand_max(t) times.
    The classical CGC literature also has a per-type LOWER bound
    (demand_min); CO-Bench's `eval_func` does not enforce it (only the
    upper bound `max` is checked), so `piece_demand_min(t)` returns 0 by
    convention -- it is still exposed for API consistency with the broader
    CGC formulation and so the LLM can choose to honor it as a "must
    include >= k" target heuristically.
  * Orientation is FIXED -- the orientation flag in every placement is 0.
  * Maximize sum of values of placed pieces.

Solution schema (matches CO-Bench's eval_func):
  {"total_value": int,
   "placements": [(piece_type_1based, x, y, length, width, orientation=0), ...]}

Tool groups:
  (1) Queries:               stock_dims, piece_dims, piece_value,
                             piece_demand_min, piece_demand_max,
                             n_piece_types
  (2) Feasibility primitives: used_count, unused_area,
                              placements_in_strip, is_guillotine_layout,
                              total_value_of
  (3) Construction:           bottom_left_pack_demand_aware,
                              guillotine_pack_BFD, try_place_piece,
                              apply_local_swap

All tools are immutable: they return new lists/dicts and do not mutate
their inputs. Coordinates are integers (CO-Bench parses ints).
"""
from __future__ import annotations

import random
import time
from typing import Iterable, Optional


def extra_tools(instance: dict) -> dict:
    """Factory: returns CGC-specific tool callables for the loaded instance.

    Instance schema (from CO-Bench load_data):
      - m:            int      number of piece types
      - stock_length: int      L of the stock
      - stock_width:  int      W of the stock
      - piece_types:  list[dict]  each {'length','width','max','value'}
    """
    m = int(instance["m"])
    L = int(instance["stock_length"])
    W = int(instance["stock_width"])
    pts = instance["piece_types"]
    # 0-indexed cached arrays for speed; piece-type IDs in solutions are 1-based.
    p_len = [int(pt["length"]) for pt in pts]
    p_wid = [int(pt["width"]) for pt in pts]
    p_max = [int(pt["max"]) for pt in pts]
    p_val = [int(pt["value"]) for pt in pts]
    # CO-Bench's eval_func does not enforce a lower demand bound -- so the
    # implicit minimum is 0. We still expose the API in case the LLM wants
    # to treat some types as "high priority".
    p_min = [0 for _ in pts]

    # ==================================================================
    # Helpers (private)
    # ==================================================================
    def _check_idx(t1: int) -> int:
        """Convert / validate 1-based piece-type id, returning 0-based index."""
        t = int(t1)
        if not (1 <= t <= m):
            raise ValueError(f"piece type {t1} out of range [1, {m}]")
        return t - 1

    def _placement_rect(p):
        """(x1,y1,x2,y2) from a 6-tuple placement."""
        x = int(p[1])
        y = int(p[2])
        return (x, y, x + int(p[3]), y + int(p[4]))

    def _overlap(r1, r2) -> bool:
        dx = min(r1[2], r2[2]) - max(r1[0], r2[0])
        dy = min(r1[3], r2[3]) - max(r1[1], r2[1])
        return dx > 0 and dy > 0

    def _is_guillotine(rects, bx, by, ex, ey) -> bool:
        """Same recursion as eval_func.is_guillotine, kept local for speed."""
        if not rects:
            return True
        if len(rects) == 1:
            r = rects[0]
            if r[0] == bx and r[1] == by and r[2] == ex and r[3] == ey:
                return True
        # candidate vertical cut positions = piece-edge x-coords strictly inside
        x_cands = sorted({r[2] for r in rects if bx < r[2] < ex} |
                         {r[0] for r in rects if bx < r[0] < ex})
        for x in x_cands:
            if all((r[2] <= x or r[0] >= x) for r in rects):
                left = [r for r in rects if r[2] <= x]
                right = [r for r in rects if r[0] >= x]
                if _is_guillotine(left, bx, by, x, ey) and \
                   _is_guillotine(right, x, by, ex, ey):
                    return True
        y_cands = sorted({r[3] for r in rects if by < r[3] < ey} |
                         {r[1] for r in rects if by < r[1] < ey})
        for y in y_cands:
            if all((r[3] <= y or r[1] >= y) for r in rects):
                bot = [r for r in rects if r[3] <= y]
                top = [r for r in rects if r[1] >= y]
                if _is_guillotine(bot, bx, by, ex, y) and \
                   _is_guillotine(top, bx, y, ex, ey):
                    return True
        return False

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def stock_dims() -> tuple:
        """(stock_length, stock_width). O(1)."""
        return (L, W)

    def piece_dims(t: int) -> tuple:
        """(length, width) of piece type `t` (1-based). O(1)."""
        i = _check_idx(t)
        return (p_len[i], p_wid[i])

    def piece_value(t: int) -> int:
        """Value of piece type `t` (1-based). O(1)."""
        return p_val[_check_idx(t)]

    def piece_demand_min(t: int) -> int:
        """Per-type lower demand bound (always 0 for CO-Bench's CGC -- no
        minimum is enforced by eval_func). Kept for API completeness so the
        LLM can choose to honor a self-imposed lower bound."""
        _check_idx(t)
        return p_min[_check_idx(t)]

    def piece_demand_max(t: int) -> int:
        """Per-type upper demand bound (the `max` count from the data).
        Exceeding this makes the solution infeasible."""
        return p_max[_check_idx(t)]

    def n_piece_types() -> int:
        """Number of distinct piece types m. O(1)."""
        return m

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def used_count(placements: Iterable, piece_type: int) -> int:
        """How many copies of `piece_type` (1-based) appear in `placements`.
        Use to check the demand_max constraint before adding another one."""
        t = int(piece_type)
        return sum(1 for p in placements if int(p[0]) == t)

    def unused_area(placements: Iterable) -> int:
        """stock_area - sum of placed piece areas. Upper bound on additional
        value-density you could still fit (does NOT account for guillotine /
        geometry). Useful as a quick prune signal."""
        used = 0
        for p in placements:
            used += int(p[3]) * int(p[4])
        return L * W - used

    def placements_in_strip(placements: Iterable, x_range: tuple) -> list:
        """All placements whose x-extent INTERSECTS [x_range[0], x_range[1]).
        Used to enumerate the pieces that would be sliced by a vertical
        guillotine cut at any x inside the range."""
        x0, x1 = int(x_range[0]), int(x_range[1])
        if x0 >= x1:
            return []
        out = []
        for p in placements:
            px0 = int(p[1])
            px1 = px0 + int(p[3])
            if px0 < x1 and px1 > x0:
                out.append(tuple(int(v) for v in p))
        return out

    def is_guillotine_layout(placements: Iterable) -> bool:
        """True iff `placements` admits a recursive edge-to-edge guillotine
        partition of the full stock. Matches CO-Bench's eval_func.is_guillotine
        (same recursion); returns False if any placement is out-of-bounds or
        overlaps another. Use as a feasibility precheck before computing
        objective."""
        rects = []
        for p in placements:
            if not (isinstance(p, (list, tuple)) and len(p) == 6):
                return False
            t = int(p[0])
            if not (1 <= t <= m):
                return False
            x, y = int(p[1]), int(p[2])
            ln, wd = int(p[3]), int(p[4])
            if int(p[5]) != 0:
                return False
            if ln != p_len[t - 1] or wd != p_wid[t - 1]:
                return False
            if x < 0 or y < 0 or x + ln > L or y + wd > W:
                return False
            rects.append((x, y, x + ln, y + wd))
        # quick pairwise-overlap check (eval_func also does this, but doing it
        # here means is_guillotine_layout fully mirrors eval_func's geometry).
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if _overlap(rects[i], rects[j]):
                    return False
        return _is_guillotine(rects, 0, 0, L, W)

    def total_value_of(placements: Iterable) -> int:
        """Sum of `value` over the piece types in `placements`. Does NOT
        check feasibility -- pair with is_guillotine_layout if needed."""
        s = 0
        for p in placements:
            t = int(p[0])
            if 1 <= t <= m:
                s += p_val[t - 1]
        return s

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def _free_rect_split(fx, fy, fL, fW, plen, pwid):
        """After placing a (plen x pwid) piece at the bottom-left of free
        rectangle (fx, fy, fL, fW), return up to two NEW free rectangles
        from a guillotine split. We always split horizontally first
        (top strip full width, then right strip of remainder). This is
        the canonical 'first horizontal cut' shelf decomposition that
        keeps the layout guillotine-feasible by construction."""
        out = []
        # top strip: same x range, above the piece
        if fW - pwid > 0:
            out.append((fx, fy + pwid, fL, fW - pwid))
        # right strip: to the right of the piece, only the piece's height
        if fL - plen > 0:
            out.append((fx + plen, fy, fL - plen, pwid))
        return out

    def _pack_in_free_rects(piece_order, demand_left):
        """Generic helper used by both construction heuristics. Maintains a
        list of axis-aligned free rectangles and tries to drop each piece
        (in the given order) into the FIRST one where it fits at (fx, fy)
        bottom-left. Honors demand_left counts. Returns a placements list
        and the (possibly updated) demand_left dict.

        The split rule (`_free_rect_split`) is a *guillotine* shelf split,
        so every layout produced here is guillotine-feasible by
        construction (no post-hoc feasibility recursion needed)."""
        placements = []
        free_rects = [(0, 0, L, W)]  # (x, y, length-extent, width-extent)
        for t in piece_order:
            idx = t - 1
            if demand_left.get(t, 0) <= 0:
                continue
            plen, pwid = p_len[idx], p_wid[idx]
            placed = False
            # try free rects in BL (lowest y, then lowest x) order
            order = sorted(range(len(free_rects)),
                           key=lambda k: (free_rects[k][1], free_rects[k][0]))
            for k in order:
                fx, fy, fL, fW = free_rects[k]
                if plen <= fL and pwid <= fW:
                    placements.append((t, fx, fy, plen, pwid, 0))
                    demand_left[t] -= 1
                    # replace this rect with its split children
                    new_rs = _free_rect_split(fx, fy, fL, fW, plen, pwid)
                    free_rects = free_rects[:k] + new_rs + free_rects[k + 1:]
                    placed = True
                    break
            if not placed:
                # leave it; maybe a smaller successor will fit somewhere
                continue
        return placements, demand_left

    def bottom_left_pack_demand_aware(piece_order: Optional[Iterable[int]] = None) -> list:
        """Bottom-left greedy with a guillotine shelf split, honoring each
        piece type's demand_max. Argument `piece_order` is an iterable of
        1-based piece-type ids -- each occurrence in the list is one
        ATTEMPT to place a copy. If omitted, an order is generated that
        flattens each type t into `demand_max(t)` copies, sorted by
        value-density (value / area) descending. Returns a placements list
        (which itself is guillotine-feasible by construction)."""
        if piece_order is None:
            order: list[int] = []
            for t in range(1, m + 1):
                order.extend([t] * p_max[t - 1])
            order.sort(key=lambda t: -(p_val[t - 1] / max(1, p_len[t - 1] * p_wid[t - 1])))
        else:
            order = [int(t) for t in piece_order]
        demand_left = {t: p_max[t - 1] for t in range(1, m + 1)}
        placements, _ = _pack_in_free_rects(order, demand_left)
        return placements

    def guillotine_pack_BFD() -> list:
        """Best-fit decreasing variant: pieces are flattened by demand_max
        and sorted by AREA descending (largest first), then dropped into
        the FIRST free rectangle that fits via the same guillotine shelf
        split as `bottom_left_pack_demand_aware`. Often beats raw BL on
        instances with mixed-size pieces. Returns a guillotine-feasible
        placements list."""
        order: list[int] = []
        for t in range(1, m + 1):
            order.extend([t] * p_max[t - 1])
        order.sort(key=lambda t: -(p_len[t - 1] * p_wid[t - 1]))
        demand_left = {t: p_max[t - 1] for t in range(1, m + 1)}
        placements, _ = _pack_in_free_rects(order, demand_left)
        return placements

    def try_place_piece(placements: Iterable, piece_type: int,
                        orient: int = 0) -> Optional[list]:
        """Attempt to add ONE copy of `piece_type` (1-based) to `placements`
        and return a NEW placements list (immutable). `orient` must be 0
        (rotation is not allowed by the problem). The function:

        1. Refuses if the piece's demand_max would be exceeded.
        2. Enumerates candidate bottom-left positions (corners of existing
           pieces and stock edges) and picks the first one where the new
           piece fits inside the stock, does not overlap any other piece,
           and the resulting full layout is still guillotine-feasible.

        Returns the new placements list on success, or None if no valid
        position exists. Useful inside iterative-improvement / repair
        loops where the LLM wants to greedily fill the residual area."""
        if int(orient) != 0:
            return None
        idx = _check_idx(piece_type)
        existing = list(tuple(int(v) for v in p) for p in placements)
        if sum(1 for p in existing if p[0] == int(piece_type)) >= p_max[idx]:
            return None
        plen, pwid = p_len[idx], p_wid[idx]
        rects = [_placement_rect(p) for p in existing]
        # candidate xs: 0 and every right edge; candidate ys: 0 and every top edge
        xs = sorted({0} | {r[2] for r in rects})
        ys = sorted({0} | {r[3] for r in rects})
        for y in ys:
            if y + pwid > W:
                continue
            for x in xs:
                if x + plen > L:
                    continue
                new_r = (x, y, x + plen, y + pwid)
                # overlap test
                bad = False
                for r in rects:
                    if _overlap(new_r, r):
                        bad = True
                        break
                if bad:
                    continue
                # full guillotine feasibility for the augmented layout
                if _is_guillotine(rects + [new_r], 0, 0, L, W):
                    return existing + [(int(piece_type), x, y, plen, pwid, 0)]
        return None

    def apply_local_swap(placements: Iterable,
                         t_limit: float = 2.0,
                         seed: Optional[int] = None) -> list:
        """Local-search pass over `placements`. Repeatedly attempts:
        (a) DROP one placement at random; (b) REPLACE it with any other
        piece type (1..m) that fits the freed area via try_place_piece
        and strictly improves total value. Keeps any strict improvement;
        reverts on non-improvement. Runs until t_limit seconds elapse or
        no improvement is found in a full pass. Returns the best
        guillotine-feasible placements list seen.

        The local move is intentionally cheap -- one drop + one insert --
        so it composes well with the construction heuristics above and
        leaves the layout guillotine-feasible at every step (try_place_piece
        enforces this)."""
        rng = random.Random(seed)
        best = list(tuple(int(v) for v in p) for p in placements)
        best_val = total_value_of(best)
        t0 = time.time()
        safety = 0.05
        improved = True
        while improved and (time.time() - t0) < float(t_limit) - safety:
            improved = False
            order = list(range(len(best)))
            rng.shuffle(order)
            for k in order:
                if (time.time() - t0) >= float(t_limit) - safety:
                    break
                dropped = best[k]
                trimmed = best[:k] + best[k + 1:]
                # Try replacing with the highest-value type that still fits.
                # Iterate by value desc so the first success is the best one
                # we have time to find.
                type_order = sorted(range(1, m + 1),
                                    key=lambda t: -p_val[t - 1])
                for t in type_order:
                    if t == dropped[0]:
                        continue  # same swap is pointless
                    cand = try_place_piece(trimmed, t)
                    if cand is None:
                        continue
                    cand_val = total_value_of(cand)
                    if cand_val > best_val:
                        best = cand
                        best_val = cand_val
                        improved = True
                        break
                if improved:
                    break
        return best

    return {
        # (1) queries
        "stock_dims": stock_dims,
        "piece_dims": piece_dims,
        "piece_value": piece_value,
        "piece_demand_min": piece_demand_min,
        "piece_demand_max": piece_demand_max,
        "n_piece_types": n_piece_types,
        # (2) feasibility primitives
        "used_count": used_count,
        "unused_area": unused_area,
        "placements_in_strip": placements_in_strip,
        "is_guillotine_layout": is_guillotine_layout,
        "total_value_of": total_value_of,
        # (3) construction
        "bottom_left_pack_demand_aware": bottom_left_pack_demand_aware,
        "guillotine_pack_BFD": guillotine_pack_BFD,
        "try_place_piece": try_place_piece,
        "apply_local_swap": apply_local_swap,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "stock_dims",
        "input": "(no args)",
        "output": "(int, int)",
        "purpose": (
            "(stock_length, stock_width) of the master stock rectangle. O(1)."
        ),
    },
    {
        "name": "piece_dims",
        "input": "t: int  (1-based piece type id)",
        "output": "(int, int)",
        "purpose": (
            "(length, width) of piece type `t`. Piece-type ids are 1-BASED "
            "throughout (matches CO-Bench's placement schema). O(1)."
        ),
    },
    {
        "name": "piece_value",
        "input": "t: int  (1-based)",
        "output": "int",
        "purpose": "Value contributed by one copy of piece type `t`. O(1).",
    },
    {
        "name": "piece_demand_min",
        "input": "t: int  (1-based)",
        "output": "int",
        "purpose": (
            "Per-type lower demand bound. CO-Bench's eval_func does NOT "
            "enforce a minimum, so this returns 0; the tool exists so you "
            "may impose a self-chosen lower bound (e.g., 'always cut at "
            "least one of type t') when designing your heuristic."
        ),
    },
    {
        "name": "piece_demand_max",
        "input": "t: int  (1-based)",
        "output": "int",
        "purpose": (
            "Per-type upper demand bound from the input data. Exceeding "
            "this in your `placements` makes the solution infeasible."
        ),
    },
    {
        "name": "n_piece_types",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of distinct piece types m. O(1).",
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "used_count",
        "input": "placements: Iterable, piece_type: int  (1-based)",
        "output": "int",
        "purpose": (
            "Count copies of `piece_type` currently in `placements`. Use to "
            "check the demand_max constraint before adding another one."
        ),
    },
    {
        "name": "unused_area",
        "input": "placements: Iterable",
        "output": "int",
        "purpose": (
            "stock_area - sum of placed-piece areas. Upper bound on extra "
            "area you might still fill; does NOT account for guillotine / "
            "fragmentation. Cheap prune signal."
        ),
    },
    {
        "name": "placements_in_strip",
        "input": "placements: Iterable, x_range: (int, int)",
        "output": "list[tuple]",
        "purpose": (
            "All placements whose x-extent intersects [x_range[0], "
            "x_range[1]). Use to enumerate the pieces that would be sliced "
            "by any vertical guillotine cut inside that range."
        ),
    },
    {
        "name": "is_guillotine_layout",
        "input": "placements: Iterable",
        "output": "bool",
        "purpose": (
            "True iff `placements` is in-bounds, overlap-free, AND admits a "
            "recursive edge-to-edge guillotine partition of the full stock. "
            "Mirrors CO-Bench's eval_func.is_guillotine. Use as a fast local "
            "feasibility check before computing objective."
        ),
    },
    {
        "name": "total_value_of",
        "input": "placements: Iterable",
        "output": "int",
        "purpose": (
            "Sum of `value` over the piece types in `placements`. No "
            "feasibility check -- pair with is_guillotine_layout if needed."
        ),
    },
    # ----- (3) Construction -----
    {
        "name": "bottom_left_pack_demand_aware",
        "input": "piece_order: Iterable[int] | None = None",
        "output": "list[placement]",
        "purpose": (
            "Greedy bottom-left construction with a GUILLOTINE shelf split. "
            "If `piece_order` is omitted, the heuristic flattens each type "
            "into demand_max copies, sorts by value-density (value / area) "
            "descending, then drops each into the first bottom-left free "
            "rectangle where it fits. Output is guillotine-feasible by "
            "construction. A great warm start: wrap as "
            "{'total_value': total_value_of(placements), 'placements': ...}."
        ),
    },
    {
        "name": "guillotine_pack_BFD",
        "input": "(no args)",
        "output": "list[placement]",
        "purpose": (
            "Best-Fit-Decreasing-by-area variant: piece copies sorted by "
            "AREA descending, then placed via the same guillotine shelf "
            "split. Output is guillotine-feasible by construction. Often "
            "complements bottom_left_pack_demand_aware (largest-first vs. "
            "highest-density-first) -- try both, keep the better."
        ),
    },
    {
        "name": "try_place_piece",
        "input": "placements: Iterable, piece_type: int, orient: int = 0",
        "output": "list[placement] | None",
        "purpose": (
            "Try to add ONE copy of `piece_type` to `placements`. Only "
            "orient=0 is valid (the problem disallows rotation). Returns a "
            "NEW placements list with the piece inserted at the first "
            "candidate corner (right-edge / top-edge of existing pieces, or "
            "the stock origin) where it fits, does not overlap, and keeps "
            "the layout guillotine-feasible; or None if no such corner "
            "exists or the demand_max would be exceeded. EXPENSIVE: each "
            "candidate triggers a guillotine recursion -- prefer "
            "bottom_left_pack_demand_aware for the bulk of construction."
        ),
    },
    {
        "name": "apply_local_swap",
        "input": "placements: Iterable, t_limit: float = 2.0, seed: int | None = None",
        "output": "list[placement]",
        "purpose": (
            "Local search over `placements`. Repeatedly: pick one placed "
            "piece, drop it, and try to replace it with a higher-value type "
            "via try_place_piece. Keeps any strict improvement; stops on "
            "stalemate or after `t_limit` seconds. Returned layout is "
            "guillotine-feasible at every step."
        ),
    },
]
