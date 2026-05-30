"""Per-problem extras for CO-Bench Packing Unequal Rectangles and Squares (AREA).

This is the AREA-maximization sibling of the "max-count" PURS task. The
solution format is identical -- a `placements` list of n tuples
(x, y, theta) with (-1, -1, 0) marking unpacked items -- but the objective
rewards bigger items more (sum of L*W of packed items, vs. just count).

The tools mirror what a packing heuristic typically needs:

  (1) Queries:        item_area, total_area, packed_indices,
                      num_packed, container_area
  (2) Geometry:       item_fits_in_container, would_overlap, aabb_of
  (3) Construction:   empty_placements, try_place,
                      greedy_by_area_first, greedy_by_count_first,
                      random_feasible_position

`try_place` returns a NEW placements list with item i set (or None if the
proposed position is infeasible) -- it is the primitive building block for
construction and local-search loops. All tools are immutable: they never
mutate their inputs.

Coordinates / rotation convention (matches eval_func):
  - Container is a circle of radius R centered at (cx, cy).
  - theta is in degrees, only 0 or 90 are accepted by eval_func; 90 swaps L<->W.
  - An item is "packed" iff (x, y) != (-1, -1) (within 1e-5 tolerance).
"""
from __future__ import annotations

import math
import random
import time
from typing import Optional

# Match eval_func's tolerances exactly so feasibility predicates here agree
# with the official checker.
_TOL = 1e-5
_ANGLE_TOL = 1e-3


def _half_dims(L: float, W: float, theta: float) -> tuple[float, float]:
    """(half_L_along_x, half_W_along_y) for axis-aligned rectangle at theta in {0,90}."""
    if abs(theta) < _ANGLE_TOL:
        return L / 2.0, W / 2.0
    if abs(theta - 90.0) < _ANGLE_TOL:
        return W / 2.0, L / 2.0
    raise ValueError(f"theta must be 0 or 90, got {theta}")


def _aabb(x: float, y: float, L: float, W: float, theta: float) -> tuple[float, float, float, float]:
    hL, hW = _half_dims(L, W, theta)
    return (x - hL, x + hL, y - hW, y + hW)


def _is_packed(p) -> bool:
    x, y, _ = p
    return not (abs(x + 1.0) < _TOL and abs(y + 1.0) < _TOL)


def extra_tools(instance: dict) -> dict:
    """Factory: returns area-PURS tool callables given the loaded instance.

    Instance schema (from CO-Bench load_data):
      - n        : int
      - cx, cy   : float, container center
      - R        : float, container radius
      - items    : list[(L, W)]
      - shape    : 'rectangle' | 'square'
      - rotation : bool, whether 90 deg rotation is allowed
    """
    n = int(instance["n"])
    cx = float(instance["cx"])
    cy = float(instance["cy"])
    R = float(instance["R"])
    items = [(float(L), float(W)) for (L, W) in instance["items"]]
    rotation_allowed = bool(instance["rotation"])
    R2 = R * R

    # Precompute areas + sort-by-area-desc index list.
    areas = [L * W for (L, W) in items]
    by_area_desc = sorted(range(n), key=lambda i: -areas[i])
    by_min_dim_desc = sorted(
        range(n), key=lambda i: -min(items[i])
    )

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def item_area(i: int) -> float:
        return float(areas[int(i)])

    def container_area() -> float:
        return math.pi * R2

    def total_area(placements) -> float:
        """Sum L*W over items considered packed (x,y != -1,-1).
        Does NOT check feasibility -- use tools['is_feasible'] for that."""
        s = 0.0
        for i, p in enumerate(placements):
            if i >= n:
                break
            if _is_packed(p):
                L, W = items[i]
                s += L * W
        return s

    def packed_indices(placements) -> list:
        return [i for i, p in enumerate(placements[:n]) if _is_packed(p)]

    def num_packed(placements) -> int:
        return sum(1 for p in placements[:n] if _is_packed(p))

    # ==================================================================
    # (2) Geometry
    # ==================================================================
    def aabb_of(i: int, x: float, y: float, theta: float = 0.0) -> tuple:
        """Axis-aligned bounding box (xmin, xmax, ymin, ymax) for item i
        placed at (x, y) with rotation theta in {0, 90}."""
        L, W = items[int(i)]
        return _aabb(float(x), float(y), L, W, float(theta))

    def item_fits_in_container(i: int, x: float, y: float, theta: float = 0.0) -> bool:
        """True iff all four corners of item i at (x, y, theta) lie inside the
        circle of radius R centered at (cx, cy). Uses the SAME tolerance as
        eval_func."""
        L, W = items[int(i)]
        try:
            hL, hW = _half_dims(L, W, float(theta))
        except ValueError:
            return False
        # The farthest corner from the center is the one diagonally opposite to
        # (cx, cy) relative to (x, y). All four corners share the same squared
        # distance from (cx, cy) only when (x, y) == (cx, cy); in general we
        # must check the four corners explicitly because (x, y) can be off
        # center. Compute the max of (|dx|+hL)^2 + (|dy|+hW)^2 ... actually
        # the corner farthest from (cx, cy) is found by going in the SAME sign
        # direction as (x-cx, y-cy) from (x, y). Equivalently:
        dx = x - cx
        dy = y - cy
        # farthest corner offsets
        fx = abs(dx) + hL
        fy = abs(dy) + hW
        return (fx * fx + fy * fy) <= R2 + _TOL

    def _aabb_overlap(a, b) -> bool:
        ox = min(a[1], b[1]) - max(a[0], b[0])
        if ox <= 0:
            return False
        oy = min(a[3], b[3]) - max(a[2], b[2])
        if oy <= 0:
            return False
        return (ox * oy) > _TOL

    def would_overlap(i: int, x: float, y: float, theta: float, placements) -> bool:
        """True iff placing item i at (x, y, theta) overlaps any already-packed
        item in `placements` (ignoring index i itself). Same overlap rule and
        tolerance as eval_func (interior-disjoint via AABBs since theta in {0,90})."""
        L, W = items[int(i)]
        try:
            a = _aabb(float(x), float(y), L, W, float(theta))
        except ValueError:
            return True  # infeasible angle -> treat as "overlap" so caller rejects
        for j, p in enumerate(placements[:n]):
            if j == i or not _is_packed(p):
                continue
            xj, yj, tj = p
            Lj, Wj = items[j]
            try:
                bj = _aabb(float(xj), float(yj), Lj, Wj, float(tj))
            except ValueError:
                continue
            if _aabb_overlap(a, bj):
                return True
        return False

    # ==================================================================
    # (3) Construction / search primitives
    # ==================================================================
    def empty_placements() -> list:
        """A fully-unpacked placements list of length n (the trivial feasible
        solution). Use as a starting point for any construction loop."""
        return [(-1.0, -1.0, 0.0) for _ in range(n)]

    def try_place(placements, i: int, x: float, y: float, theta: float = 0.0):
        """If item i can be placed at (x, y, theta) without leaving the
        container or overlapping any other packed item, return a NEW
        placements list with that slot updated. Otherwise return None.

        Respects the instance's `rotation` flag (theta must be 0 when
        rotation is False)."""
        ii = int(i)
        if not (0 <= ii < n):
            return None
        th = float(theta)
        if not rotation_allowed and abs(th) > _ANGLE_TOL:
            return None
        if not (abs(th) < _ANGLE_TOL or abs(th - 90.0) < _ANGLE_TOL):
            return None
        if not item_fits_in_container(ii, x, y, th):
            return None
        if would_overlap(ii, x, y, th, placements):
            return None
        out = list(placements)
        # pad if too short
        while len(out) < n:
            out.append((-1.0, -1.0, 0.0))
        out[ii] = (float(x), float(y), th)
        return out

    def _candidate_positions(i: int, theta: float, placements, grid: int = 21) -> list:
        """Yield (x, y) candidate centers for item i on a square grid that
        spans the circle's bounding box, intersected with the inside-circle
        feasibility test. Coarse-to-fine grid, snapped to corners of already-
        placed items for tighter packing."""
        L, W = items[int(i)]
        try:
            hL, hW = _half_dims(L, W, float(theta))
        except ValueError:
            return []
        # legal center box (inscribed rectangle of half-extents (R-hL, R-hW))
        lo_x = cx - max(0.0, R - hL)
        hi_x = cx + max(0.0, R - hL)
        lo_y = cy - max(0.0, R - hW)
        hi_y = cy + max(0.0, R - hW)
        if hi_x < lo_x or hi_y < lo_y:
            return []
        cands = []
        if grid >= 2:
            for ix in range(grid):
                tx = lo_x if grid == 1 else lo_x + (hi_x - lo_x) * ix / (grid - 1)
                for iy in range(grid):
                    ty = lo_y if grid == 1 else lo_y + (hi_y - lo_y) * iy / (grid - 1)
                    cands.append((tx, ty))
        # also try snapping just outside each existing packed item's AABB so
        # we lay rectangles flush against neighbors (helps area packings)
        for j, p in enumerate(placements[:n]):
            if not _is_packed(p):
                continue
            xj, yj, tj = p
            Lj, Wj = items[j]
            try:
                bj = _aabb(float(xj), float(yj), Lj, Wj, float(tj))
            except ValueError:
                continue
            xmin, xmax, ymin, ymax = bj
            # 4 sides: place new item flush right / left / above / below
            cands.append((xmax + hL, yj))
            cands.append((xmin - hL, yj))
            cands.append((xj, ymax + hW))
            cands.append((xj, ymin - hW))
        return cands

    def _greedy_by_order(order, time_limit_s: float, grid: int) -> list:
        """Shared core: walk items in the given order, for each try (0, 90)
        rotations and a grid of candidate centers; place at the first
        feasible (best-fit-ish) position."""
        out = [(-1.0, -1.0, 0.0) for _ in range(n)]
        t0 = time.time()
        safety = 0.05
        for i in order:
            if (time.time() - t0) >= max(0.0, time_limit_s - safety):
                break
            thetas = [0.0]
            if rotation_allowed:
                thetas.append(90.0)
            placed = False
            for th in thetas:
                if placed:
                    break
                cands = _candidate_positions(i, th, out, grid=grid)
                for (px, py) in cands:
                    if not item_fits_in_container(i, px, py, th):
                        continue
                    if would_overlap(i, px, py, th, out):
                        continue
                    out[i] = (float(px), float(py), float(th))
                    placed = True
                    break
        return out

    def greedy_by_area_first(time_limit_s: float = 5.0, grid: int = 21) -> list:
        """Greedy construction that tries to place items in DECREASING AREA
        order on a `grid`x`grid` candidate-center lattice (plus flush-against-
        neighbor snap points). Returns a feasible placements list (some items
        may be left unpacked). For the area objective this is usually the
        single best warm start: bigger items are scarcer and worth more."""
        return _greedy_by_order(by_area_desc, time_limit_s, grid)

    def greedy_by_count_first(time_limit_s: float = 5.0, grid: int = 21) -> list:
        """Same greedy loop but with items in INCREASING area order (small
        first). Tends to pack more items but lower total area -- useful as a
        diversification baseline / for hybrid strategies."""
        order = sorted(range(n), key=lambda i: areas[i])
        return _greedy_by_order(order, time_limit_s, grid)

    def random_feasible_position(
        i: int,
        theta: float = 0.0,
        placements=None,
        max_tries: int = 200,
        rng: Optional[random.Random] = None,
    ):
        """Sample up to `max_tries` random centers (uniform in the inscribed
        legal box) and return the first (x, y) where item i with rotation
        `theta` is feasible given `placements`. Returns None if no feasible
        sample was found. Respects the instance's `rotation` flag."""
        if rng is None:
            rng = random
        if placements is None:
            placements = empty_placements()
        if not rotation_allowed and abs(float(theta)) > _ANGLE_TOL:
            return None
        L, W = items[int(i)]
        try:
            hL, hW = _half_dims(L, W, float(theta))
        except ValueError:
            return None
        lo_x, hi_x = cx - max(0.0, R - hL), cx + max(0.0, R - hL)
        lo_y, hi_y = cy - max(0.0, R - hW), cy + max(0.0, R - hW)
        if hi_x < lo_x or hi_y < lo_y:
            return None
        for _ in range(int(max_tries)):
            x = rng.uniform(lo_x, hi_x)
            y = rng.uniform(lo_y, hi_y)
            if not item_fits_in_container(i, x, y, theta):
                continue
            if would_overlap(i, x, y, float(theta), placements):
                continue
            return (x, y)
        return None

    return {
        # queries
        "item_area": item_area,
        "container_area": container_area,
        "total_area": total_area,
        "packed_indices": packed_indices,
        "num_packed": num_packed,
        # geometry
        "aabb_of": aabb_of,
        "item_fits_in_container": item_fits_in_container,
        "would_overlap": would_overlap,
        # construction
        "empty_placements": empty_placements,
        "try_place": try_place,
        "greedy_by_area_first": greedy_by_area_first,
        "greedy_by_count_first": greedy_by_count_first,
        "random_feasible_position": random_feasible_position,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "item_area",
        "input": "i: int",
        "output": "float",
        "purpose": (
            "Area L*W of item i (zero-indexed). The OBJECTIVE is the sum of "
            "item_area(i) over packed items, so bigger items are worth more."
        ),
    },
    {
        "name": "container_area",
        "input": "(no args)",
        "output": "float",
        "purpose": (
            "pi * R^2 -- a trivial upper bound on the achievable total area. "
            "Useful as a sanity check / early stopping threshold."
        ),
    },
    {
        "name": "total_area",
        "input": "placements: list",
        "output": "float",
        "purpose": (
            "Sum of L*W over items whose placement is not (-1, -1). Does NOT "
            "check feasibility (use tools['is_feasible'] for that). Much faster "
            "than tools['objective'] when you just want to score a candidate "
            "during local search."
        ),
    },
    {
        "name": "packed_indices",
        "input": "placements: list",
        "output": "list[int]",
        "purpose": "Indices of items considered packed (i.e., x != -1 or y != -1).",
    },
    {
        "name": "num_packed",
        "input": "placements: list",
        "output": "int",
        "purpose": "Count of packed items in the placements list.",
    },
    # ----- Geometry -----
    {
        "name": "aabb_of",
        "input": "i: int, x: float, y: float, theta: float = 0.0",
        "output": "(xmin, xmax, ymin, ymax)",
        "purpose": (
            "Axis-aligned bounding box of item i if placed at (x, y, theta). "
            "Since theta is restricted to {0, 90}, the AABB is exact."
        ),
    },
    {
        "name": "item_fits_in_container",
        "input": "i: int, x: float, y: float, theta: float = 0.0",
        "output": "bool",
        "purpose": (
            "True iff every corner of item i (with rotation theta in {0, 90}) "
            "lies inside the circular container, using the same tolerance as "
            "eval_func."
        ),
    },
    {
        "name": "would_overlap",
        "input": "i: int, x: float, y: float, theta: float, placements: list",
        "output": "bool",
        "purpose": (
            "True iff placing item i at (x, y, theta) overlaps any already-"
            "packed item in `placements` (ignoring slot i itself). Mirrors "
            "eval_func's interior-disjoint check."
        ),
    },
    # ----- Construction / search -----
    {
        "name": "empty_placements",
        "input": "(no args)",
        "output": "list[(float, float, float)]",
        "purpose": (
            "Length-n list of (-1, -1, 0) tuples -- the trivial feasible "
            "solution with zero packed items. Starting point for any "
            "construction loop."
        ),
    },
    {
        "name": "try_place",
        "input": "placements: list, i: int, x: float, y: float, theta: float = 0.0",
        "output": "list | None",
        "purpose": (
            "If item i fits at (x, y, theta) without leaving the container or "
            "overlapping any packed item, return a NEW placements list with "
            "that slot updated. Otherwise return None. Respects the instance's "
            "`rotation` flag (theta must be 0 when rotation is disallowed). "
            "Primitive building block for greedy and local-search loops."
        ),
    },
    {
        "name": "greedy_by_area_first",
        "input": "time_limit_s: float = 5.0, grid: int = 21",
        "output": "list[(x, y, theta)]",
        "purpose": (
            "Greedy construction in DECREASING-AREA order: for each item try "
            "rotations {0} or {0, 90} and a grid x grid lattice of candidate "
            "centers PLUS snap points flush against already-placed neighbors. "
            "Returns a feasible placements list (some items may stay unpacked). "
            "Typically the best single warm start for the area objective -- "
            "bigger items matter more, so place them first."
        ),
    },
    {
        "name": "greedy_by_count_first",
        "input": "time_limit_s: float = 5.0, grid: int = 21",
        "output": "list[(x, y, theta)]",
        "purpose": (
            "Same greedy as greedy_by_area_first but in INCREASING-area order "
            "(small items first). Packs more items in total but usually less "
            "total area; useful as a diversification baseline or for hybrid "
            "strategies (e.g., big-first then squeeze in small)."
        ),
    },
    {
        "name": "random_feasible_position",
        "input": "i: int, theta: float = 0.0, placements: list = None, max_tries: int = 200, rng: random.Random = None",
        "output": "(x, y) | None",
        "purpose": (
            "Sample up to `max_tries` uniformly-random centers inside the "
            "circle's legal box and return the first (x, y) where item i at "
            "rotation theta is feasible w.r.t. `placements`. Returns None if "
            "none found. Use for randomized multi-start / ILS perturbations."
        ),
    },
]
