"""Per-problem extras for CO-Bench 'Packing unequal rectangles and squares'.

Provides primitive building blocks so the LLM can compose construction +
local-search heuristics for 2-D circular-container packing without
re-deriving bottom-left / corner-fit / overlap tests from scratch.

The underlying problem (see config.py):
  - Container: circle of radius R centered at (cx, cy).
  - Items:     n axis-aligned rectangles (squares iff L==W), given as (L, W).
  - Rotation:  if instance['rotation'] is True, an item may be rotated 90 deg
               (its L and W are swapped). Squares trivially ignore rotation.
  - Goal:      MAXIMIZE the number of packed items.

CO-Bench solution schema (passed back as the 'placements' key):
    [(x_0, y_0, theta_0), (x_1, y_1, theta_1), ..., (x_{n-1}, y_{n-1}, theta_{n-1})]
  - exactly n tuples, one per item (0-indexed, matching `items`);
  - theta in {0, 90}: 90 only when rotation is allowed (else MUST be 0);
  - unpacked items use (-1, -1, 0).

These extras use a parallel internal representation throughout:
    placements: dict[int, (x, y, theta)]
mapping packed-item-index -> placement. Unpacked items are simply absent from
the dict. Use `placements_to_solution(placements)` to expand a dict into the
n-tuple list CO-Bench expects.

Tool groups:
  (1) Queries:           item_dims, container_dims, n_items
  (2) Geometry checks:   rects_overlap, is_inside_container, can_fit_at,
                         placements_to_solution
  (3) Construction /
      local search:      bottom_left_pack, bottom_left_fill_decreasing,
                         try_place_largest_unplaced, apply_swap_items

All tools are immutable: they return new dicts/lists and do not mutate inputs.
"""
from __future__ import annotations

import math
import time
from typing import Iterable, Optional

# Geometry tolerance. CO-Bench's eval_func uses tol=1e-5 for both the
# corner-inside-circle check and the no-overlap check, so we mirror that
# constant here -- candidate placements must clear these by at least `_EPS`.
_EPS = 1e-5


def extra_tools(instance: dict) -> dict:
    """Factory: returns Packing-specific tool callables for the loaded instance.

    Instance schema (from CO-Bench load_data):
      - n:         int                 -- number of items
      - cx, cy:    float               -- container center
      - R:         float               -- container radius
      - items:     list[(L, W)]        -- 0-indexed item dimensions
      - shape:     'rectangle' | 'square'
      - rotation:  bool                -- True iff 90 deg rotation is allowed
    """
    n = int(instance["n"])
    cx = float(instance["cx"])
    cy = float(instance["cy"])
    R = float(instance["R"])
    items = [(float(L), float(W)) for (L, W) in instance["items"]]
    if len(items) != n:
        n = len(items)
    shape = str(instance.get("shape", "rectangle")).lower()
    rotation_allowed = bool(instance.get("rotation", False))

    # Precompute item areas; used by area-desc construction heuristics.
    areas = [L * W for (L, W) in items]
    # Sort by area DESCENDING -- bottom-left-fill-decreasing places big items first.
    order_by_area_desc = sorted(range(n), key=lambda i: -areas[i])

    # ==================================================================
    # Internal helpers (closure-private)
    # ==================================================================
    def _eff_dims(idx: int, theta: float) -> tuple[float, float]:
        """Effective (L, W) of an item given its rotation theta (0 or 90)."""
        L, W = items[idx]
        if math.isclose(float(theta), 90.0, abs_tol=1e-3):
            return (W, L)
        return (L, W)

    def _bbox(placement: tuple, idx: int) -> tuple[float, float, float, float]:
        """Axis-aligned (xmin, xmax, ymin, ymax) of item `idx` at `placement`."""
        x, y, theta = placement
        eL, eW = _eff_dims(idx, theta)
        return (x - eL / 2.0, x + eL / 2.0, y - eW / 2.0, y + eW / 2.0)

    def _corners_inside(bbox: tuple[float, float, float, float]) -> bool:
        """True iff every corner of `bbox` lies inside the circle (with eps slack).
        CO-Bench's eval_func tolerates corners up to R + 1e-5, so we accept any
        corner whose distance squared is at most R^2 plus a tiny safety pad."""
        xmin, xmax, ymin, ymax = bbox
        # Worst-case corner is the one farthest from (cx, cy). For an
        # axis-aligned bbox that's the corner whose |dx|,|dy| are largest.
        dx = max(abs(xmin - cx), abs(xmax - cx))
        dy = max(abs(ymin - cy), abs(ymax - cy))
        return (dx * dx + dy * dy) <= R * R + 1e-9

    def _bbox_overlap(a: tuple, b: tuple) -> bool:
        """True iff two axis-aligned bboxes (xmin, xmax, ymin, ymax) overlap
        with non-empty interior (CO-Bench's strict overlap rule uses a
        positive tolerance; we require a gap of at least _EPS to be safe)."""
        axmin, axmax, aymin, aymax = a
        bxmin, bxmax, bymin, bymax = b
        if axmax <= bxmin + _EPS or bxmax <= axmin + _EPS:
            return False
        if aymax <= bymin + _EPS or bymax <= aymin + _EPS:
            return False
        return True

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def item_dims(i: int) -> tuple[float, float]:
        """Original (L, W) of item `i` (0-indexed). Rotation does NOT change
        what this returns -- use _eff_dims via can_fit_at / bbox-aware tools
        if you need the rotated dims."""
        if not (0 <= int(i) < n):
            raise ValueError(f"item index {i} out of range [0, {n})")
        return items[int(i)]

    def container_dims() -> tuple[float, float, float]:
        """(cx, cy, R) of the circular container."""
        return (cx, cy, R)

    def n_items() -> int:
        """Total number of items n in this instance."""
        return n

    # ==================================================================
    # (2) Geometry checks
    # ==================================================================
    def rects_overlap(p1: tuple, p2: tuple, idx1: int, idx2: int) -> bool:
        """True iff items `idx1`,`idx2` placed at `p1`,`p2` (each a
        (x, y, theta) tuple) overlap with non-empty interior. Uses the
        same tolerance as CO-Bench's eval_func."""
        return _bbox_overlap(_bbox(p1, idx1), _bbox(p2, idx2))

    def is_inside_container(placement: tuple, idx: int) -> bool:
        """True iff item `idx` at `placement` has all four corners strictly
        inside the circular container (with a small safety margin so the
        eval-time check, which permits R + 1e-5, is always satisfied)."""
        return _corners_inside(_bbox(placement, idx))

    def can_fit_at(placements: dict, item_idx: int, x: float, y: float,
                   theta: float = 0.0) -> bool:
        """True iff item `item_idx` can be placed at (x, y, theta) without:
          (a) violating rotation rules (theta must be 0; or 0/90 if allowed),
          (b) extending outside the circular container, or
          (c) overlapping any item already in `placements`.

        `placements` is a dict[int, (x, y, theta)] mapping item index to
        its current placement. Items not in the dict are treated as
        unplaced (and therefore irrelevant for overlap)."""
        # rotation rule
        if not rotation_allowed and not math.isclose(theta, 0.0, abs_tol=1e-3):
            return False
        if rotation_allowed and not (
            math.isclose(theta, 0.0, abs_tol=1e-3)
            or math.isclose(theta, 90.0, abs_tol=1e-3)
        ):
            return False
        cand = (float(x), float(y), float(theta))
        cand_box = _bbox(cand, int(item_idx))
        if not _corners_inside(cand_box):
            return False
        for j, p in placements.items():
            if int(j) == int(item_idx):
                continue
            if _bbox_overlap(cand_box, _bbox(p, int(j))):
                return False
        return True

    def placements_to_solution(placements: dict) -> dict:
        """Convert an internal dict[int, (x, y, theta)] into the CO-Bench
        solution dict: {'placements': [(x, y, theta), ...]} with exactly n
        entries. Items missing from `placements` are emitted as (-1, -1, 0)
        which CO-Bench interprets as 'unpacked'."""
        out: list[tuple] = []
        for i in range(n):
            if i in placements:
                x, y, theta = placements[i]
                out.append((float(x), float(y), float(theta)))
            else:
                out.append((-1, -1, 0))
        return {"placements": out}

    # ==================================================================
    # (3) Construction / local search
    # ==================================================================
    def _candidate_points(placed_bboxes: list[tuple]) -> list[tuple[float, float]]:
        """Bottom-left corner candidate set. The classic BL-fill heuristic
        considers points where a new item's bottom-left can sit:
          - the bottom-left of the INSCRIBED-SQUARE seed (inside the circle),
          - corners of each already-placed bbox (so new items flush against
            existing ones), and
          - a coarse grid of points along the left edge of the inscribed
            region (helps the very first / very large items find a slot).
        The caller still tests each candidate with the circular-containment
        and overlap rules, so candidates outside the disk are filtered out."""
        pts: list[tuple[float, float]] = []
        # Inscribed-square seed: the largest axis-aligned square inside the
        # circle has half-side R/sqrt(2). Its bottom-left corner (cx - s, cy - s)
        # is the natural bottom-left seed for the first item (still inside the
        # disk, unlike (cx - R, cy - R) which sits outside).
        s = R / math.sqrt(2.0)
        pts.append((cx - s, cy - s))
        # Coarse grid along the bottom-left arc of the disk: helps seed
        # large rectangles that wouldn't fit at any other corner yet.
        # ~7 points is enough; we only need ONE feasible spot per item.
        for k in range(7):
            ty = cy - s + (2 * s) * (k / 6.0)  # span from cy-s to cy+s
            # leftmost x with this y inside the circle
            dy = ty - cy
            if R * R - dy * dy < 0:
                continue
            tx = cx - math.sqrt(R * R - dy * dy)
            pts.append((tx, ty))
        for (xmin, xmax, ymin, ymax) in placed_bboxes:
            # right side of placed bbox (new item's left edge sits here)
            pts.append((xmax, ymin))
            pts.append((xmax, ymax))
            # top side of placed bbox (new item's bottom edge sits here)
            pts.append((xmin, ymax))
            pts.append((xmax, ymax))
            # bottom-left corner too (covers L-shaped gaps where the new
            # item sits flush below-and-left of an existing one)
            pts.append((xmin, ymin))
        # dedupe (rounded) so we don't try identical points repeatedly
        seen = set()
        out = []
        for (px, py) in pts:
            key = (round(px, 6), round(py, 6))
            if key not in seen:
                seen.add(key)
                out.append((px, py))
        return out

    def bottom_left_pack(item_order: Iterable[int]) -> dict:
        """Greedy bottom-left construction following the given `item_order`.

        For each item in order, scan a candidate-corner set (built from the
        bboxes of items already placed, plus a seed at the container's
        bottom-left). For each candidate corner (cx_pt, cy_pt) treated as
        the item's BOTTOM-LEFT, compute the implied center, try both
        rotations (if allowed), and accept the first feasible placement.
        Items that cannot be placed are simply omitted from the returned
        dict (i.e., remain 'unpacked' in the final solution).

        Returns: dict[int, (x, y, theta)]  -- a partial placement map.
        Use placements_to_solution(...) to convert to CO-Bench's format."""
        placements: dict = {}
        placed_bboxes: list[tuple] = []
        for idx in item_order:
            idx = int(idx)
            if not (0 <= idx < n):
                continue
            best_placement = None
            best_score: Optional[tuple] = None  # (py, px) -- lower is better
            cand_points = _candidate_points(placed_bboxes)
            thetas = [0.0, 90.0] if rotation_allowed else [0.0]
            # squares: rotation is redundant
            L, W = items[idx]
            if math.isclose(L, W, abs_tol=1e-9):
                thetas = [0.0]
            for theta in thetas:
                eL, eW = _eff_dims(idx, theta)
                for (px, py) in cand_points:
                    # (px, py) interpreted as bottom-left of the item.
                    cx_item = px + eL / 2.0
                    cy_item = py + eW / 2.0
                    cand = (cx_item, cy_item, theta)
                    cand_box = _bbox(cand, idx)
                    if not _corners_inside(cand_box):
                        continue
                    bad = False
                    for b in placed_bboxes:
                        if _bbox_overlap(cand_box, b):
                            bad = True
                            break
                    if bad:
                        continue
                    score = (py, px)  # primary: bottom-most; tiebreak: left-most
                    if best_placement is None or score < best_score:
                        best_placement = cand
                        best_score = score
            if best_placement is not None:
                placements[idx] = best_placement
                placed_bboxes.append(_bbox(best_placement, idx))
        return placements

    def bottom_left_fill_decreasing() -> dict:
        """Convenience: bottom_left_pack with items ordered by AREA descending.
        Mirrors the classic FFD strategy from 1-D bin packing -- place big
        items first so small items fit in the leftover gaps. Returns the
        same dict[int, (x, y, theta)] as bottom_left_pack."""
        return bottom_left_pack(list(order_by_area_desc))

    def try_place_largest_unplaced(placements: dict) -> dict:
        """Single-shot improvement: among items currently NOT in `placements`,
        find the LARGEST (by area) and try to place it at any candidate
        bottom-left corner of the existing layout. Returns a NEW placements
        dict if a feasible spot is found, else returns the input unchanged.

        Useful as a 'one more item' polish step after bottom_left_pack."""
        unplaced = [i for i in range(n) if i not in placements]
        if not unplaced:
            return placements
        # sort unplaced largest-first
        unplaced.sort(key=lambda i: -areas[i])
        placed_bboxes = [_bbox(p, i) for i, p in placements.items()]
        thetas_default = [0.0, 90.0] if rotation_allowed else [0.0]
        for idx in unplaced:
            L, W = items[idx]
            thetas = [0.0] if math.isclose(L, W, abs_tol=1e-9) else thetas_default
            cand_points = _candidate_points(placed_bboxes)
            for theta in thetas:
                eL, eW = _eff_dims(idx, theta)
                for (px, py) in cand_points:
                    cx_item = px + eL / 2.0
                    cy_item = py + eW / 2.0
                    cand = (cx_item, cy_item, theta)
                    cand_box = _bbox(cand, idx)
                    if not _corners_inside(cand_box):
                        continue
                    bad = any(_bbox_overlap(cand_box, b) for b in placed_bboxes)
                    if bad:
                        continue
                    # success -- return a NEW dict
                    new_p = dict(placements)
                    new_p[idx] = cand
                    return new_p
        return placements

    def apply_swap_items(placements: dict, time_limit_s: float = 5.0) -> dict:
        """Local search: try to REPLACE one placed item with one or more
        currently-unplaced items, aiming to grow |placements|. The strategy:

          for each currently placed item p (smallest-first; cheapest to lose):
            tentatively remove it, then re-run a bottom-left fill that prefers
            unplaced items by AREA descending. If the result has MORE items
            than the original layout, keep it.

        This is a 'remove-1, add-many' move -- the standard improvement step
        for max-count 2-D packing. First-improvement, restarts the outer loop
        whenever a swap is accepted; stops when time_limit_s elapses or no
        further swap helps.

        Returns a NEW placements dict (immutable interface)."""
        cur = dict(placements)
        t0 = time.time()
        safety = 0.05
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            # try removing the smallest placed items first (cheapest sacrifice)
            placed_ids = sorted(cur.keys(), key=lambda i: areas[i])
            for victim in placed_ids:
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                trial = dict(cur)
                del trial[victim]
                # Re-pack ALL currently-unplaced items (including the victim)
                # in area-desc order on top of the surviving layout.
                unplaced = [i for i in range(n) if i not in trial]
                unplaced.sort(key=lambda i: -areas[i])
                placed_bboxes = [_bbox(p, i) for i, p in trial.items()]
                # try to add each in turn (greedy bottom-left)
                for idx in unplaced:
                    L, W = items[idx]
                    thetas = ([0.0] if math.isclose(L, W, abs_tol=1e-9)
                              else ([0.0, 90.0] if rotation_allowed else [0.0]))
                    cand_points = _candidate_points(placed_bboxes)
                    chosen = None
                    chosen_score = None
                    for theta in thetas:
                        eL, eW = _eff_dims(idx, theta)
                        for (px, py) in cand_points:
                            cx_item = px + eL / 2.0
                            cy_item = py + eW / 2.0
                            cand = (cx_item, cy_item, theta)
                            cand_box = _bbox(cand, idx)
                            if not _corners_inside(cand_box):
                                continue
                            if any(_bbox_overlap(cand_box, b) for b in placed_bboxes):
                                continue
                            score = (py, px)
                            if chosen is None or score < chosen_score:
                                chosen = cand
                                chosen_score = score
                    if chosen is not None:
                        trial[idx] = chosen
                        placed_bboxes.append(_bbox(chosen, idx))
                if len(trial) > len(cur):
                    cur = trial
                    improved = True
                    break  # restart outer while-loop with the new layout
        return cur

    return {
        # (1) queries
        "item_dims": item_dims,
        "container_dims": container_dims,
        "n_items": n_items,
        # (2) geometry checks
        "rects_overlap": rects_overlap,
        "is_inside_container": is_inside_container,
        "can_fit_at": can_fit_at,
        "placements_to_solution": placements_to_solution,
        # (3) construction / local search
        "bottom_left_pack": bottom_left_pack,
        "bottom_left_fill_decreasing": bottom_left_fill_decreasing,
        "try_place_largest_unplaced": try_place_largest_unplaced,
        "apply_swap_items": apply_swap_items,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "item_dims",
        "input": "i: int  (0-based item id)",
        "output": "tuple[float, float]  (L, W)",
        "purpose": (
            "Original (L, W) dimensions of item `i` (0-indexed; matches "
            "instance['items'] order, which CO-Bench sorts by increasing "
            "size). For squares L == W. Rotation does not change what this "
            "returns -- it always reports the unrotated dimensions. O(1)."
        ),
    },
    {
        "name": "container_dims",
        "input": "(no args)",
        "output": "tuple[float, float, float]  (cx, cy, R)",
        "purpose": (
            "Center (cx, cy) and radius R of the circular container. "
            "Useful for sampling candidate positions inside the disk. O(1)."
        ),
    },
    {
        "name": "n_items",
        "input": "(no args)",
        "output": "int",
        "purpose": "Total number of available items n. O(1).",
    },
    # ----- (2) Geometry checks -----
    {
        "name": "rects_overlap",
        "input": "p1: (x, y, theta), p2: (x, y, theta), idx1: int, idx2: int",
        "output": "bool",
        "purpose": (
            "True iff items `idx1`, `idx2` placed at `p1`, `p2` overlap with "
            "non-empty interior. Mirrors CO-Bench's overlap test (axis-aligned "
            "bbox, strict inequality with eps slack). O(1)."
        ),
    },
    {
        "name": "is_inside_container",
        "input": "placement: (x, y, theta), idx: int",
        "output": "bool",
        "purpose": (
            "True iff all four corners of item `idx` placed at `placement` "
            "lie inside the circle. Uses a small safety margin below R so the "
            "eval-time tolerance (R + 1e-5) is always satisfied. O(1)."
        ),
    },
    {
        "name": "can_fit_at",
        "input": ("placements: dict[int, (x, y, theta)], item_idx: int, "
                  "x: float, y: float, theta: float = 0.0"),
        "output": "bool",
        "purpose": (
            "True iff item `item_idx` can be placed at (x, y, theta) without "
            "violating the rotation rule (theta=0 always; theta=90 only if "
            "rotation allowed), without extending outside the circle, and "
            "without overlapping any item already in `placements`. The cheap "
            "primitive for neighborhood search."
        ),
    },
    {
        "name": "placements_to_solution",
        "input": "placements: dict[int, (x, y, theta)]",
        "output": "dict  {'placements': list[(x, y, theta)]}",
        "purpose": (
            "Convert an internal dict (only PLACED items) into CO-Bench's "
            "n-tuple list, filling missing items with (-1, -1, 0) to mark "
            "them unpacked. ALWAYS use this before returning your solution -- "
            "CO-Bench requires exactly n placements."
        ),
    },
    # ----- (3) Construction / local search -----
    {
        "name": "bottom_left_pack",
        "input": "item_order: Iterable[int]  (0-based item indices)",
        "output": "dict[int, (x, y, theta)]",
        "purpose": (
            "Greedy bottom-left construction in the given `item_order`. For "
            "each item, scan a candidate-corner set built from the bboxes of "
            "already-placed items (plus a seed at the container bottom-left), "
            "treat each candidate as the item's BOTTOM-LEFT, try both "
            "rotations if allowed, and accept the most bottom-left feasible "
            "placement. Items that don't fit are silently skipped (left "
            "unpacked). O(n^2) candidate corners times O(n) overlap test."
        ),
    },
    {
        "name": "bottom_left_fill_decreasing",
        "input": "(no args)",
        "output": "dict[int, (x, y, theta)]",
        "purpose": (
            "Bottom-left-fill with items ORDERED BY AREA DESCENDING. "
            "Equivalent to bottom_left_pack(sorted(range(n), key=area desc)). "
            "Strong default warm start: place big items first so small items "
            "fall into leftover gaps. Often within a small additive gap of "
            "the optimum on the CO-Bench reference cases."
        ),
    },
    {
        "name": "try_place_largest_unplaced",
        "input": "placements: dict[int, (x, y, theta)]",
        "output": "dict[int, (x, y, theta)]",
        "purpose": (
            "Polish step: try to insert the largest currently-unplaced item "
            "into the existing layout at any feasible bottom-left corner. "
            "Returns a NEW placements dict (with the new item added) on "
            "success, else returns the input dict unchanged. Cheap; call "
            "repeatedly until it returns the same dict."
        ),
    },
    {
        "name": "apply_swap_items",
        "input": "placements: dict[int, (x, y, theta)], time_limit_s: float = 5.0",
        "output": "dict[int, (x, y, theta)]",
        "purpose": (
            "Local-search 'remove-1, add-many' improvement loop. For each "
            "placed item (smallest area first), tentatively remove it, then "
            "greedily fill any unplaced items into the freed space using "
            "bottom-left fill (area-desc). Keep the swap iff |placements| "
            "grew. First-improvement; restarts after every accepted swap; "
            "stops at time_limit_s. Returns a NEW dict."
        ),
    },
]
