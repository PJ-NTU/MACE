"""Per-problem extras for CO-Bench Packing Unequal Circles (maximize-number).

The task: given a CIRCULAR container of radius R centered at (cx, cy) and
a list of n circles with radii sorted in *non-decreasing* order, pack as
many as possible. CO-Bench enforces a PREFIX constraint -- if circle i is
packed, then every circle with smaller index must also be packed. Because
radii are sorted ascending, the optimal strategy almost always packs the
SMALLEST circles first (they are easiest to fit).

Tools fall in 3 tiers (no Tier-4 exact solver -- continuous non-convex
packing has no scalable exact method):

  (1) Queries:
        circle_radius(i), container_dims(), n_circles()
  (2) Feasibility primitives:
        is_inside_container(x, y, r),
        circles_overlap(c1_xyr, c2_xyr),
        space_for_circle(placements, r, x, y)
  (3) Construction / improvement:
        prefix_grid_construct(grid_steps),
        front_packing_construct(),
        apply_local_shift(placements, t_limit_s),
        try_place_next(placements)

Solution schema (CO-Bench):
    {"coords": [(x, y), ...]}   length n; unpacked circles use (-1, -1).
The prefix property MUST hold: packed indices = [0, 1, ..., K] for some K.

All tools work in immutable style: they take and return placements as
list[(x, y) or None] where index i is circle i's center (or None if not
yet placed). The conversion to the CO-Bench `coords` list (None -> (-1,-1))
is the caller's responsibility, but `to_coords(placements)` helps.
"""
from __future__ import annotations

import math
import random
import time
from typing import Optional


_EPS = 1e-7  # CO-Bench tolerance is 1e-5; we stay well inside it.


def extra_tools(instance: dict) -> dict:
    """Factory: returns Packing-Unequal-Circles tool callables.

    Instance schema (from CO-Bench load_data):
      - n     : int
      - cx,cy : float    container center
      - R     : float    container radius
      - radii : list[float]   sorted non-decreasing
    """
    n = int(instance["n"])
    cx = float(instance["cx"])
    cy = float(instance["cy"])
    R = float(instance["R"])
    radii = [float(r) for r in instance["radii"]]

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def n_circles() -> int:
        """Total number of circles n."""
        return n

    def circle_radius(i: int) -> float:
        """Radius of circle i (0-based)."""
        if not (0 <= int(i) < n):
            raise ValueError(f"circle index {i} out of range [0, {n})")
        return radii[int(i)]

    def container_dims() -> dict:
        """Container geometry as {'shape': 'circle', 'cx', 'cy', 'R'}."""
        return {"shape": "circle", "cx": cx, "cy": cy, "R": R}

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def is_inside_container(x: float, y: float, r: float) -> bool:
        """True iff circle (x, y, r) lies fully inside the container, i.e.
        dist((x,y), (cx,cy)) + r <= R (with tolerance)."""
        d = math.hypot(x - cx, y - cy)
        return d + float(r) <= R + _EPS

    def circles_overlap(c1_xyr, c2_xyr) -> bool:
        """True iff two circles overlap (interiors intersect). Each input is
        a 3-tuple (x, y, r). Tangent contact counts as non-overlapping."""
        x1, y1, r1 = float(c1_xyr[0]), float(c1_xyr[1]), float(c1_xyr[2])
        x2, y2, r2 = float(c2_xyr[0]), float(c2_xyr[1]), float(c2_xyr[2])
        d = math.hypot(x1 - x2, y1 - y2)
        return d + _EPS < r1 + r2

    def space_for_circle(placements, r: float, x: float, y: float) -> bool:
        """True iff placing a circle of radius `r` at (x, y) is feasible
        given `placements`: (a) inside container, (b) no overlap with any
        already-placed circle in `placements`.

        `placements` is a list of length n where placements[i] is the (x,y)
        center of circle i if it has been placed, else None.
        """
        if not is_inside_container(x, y, r):
            return False
        for j, p in enumerate(placements):
            if p is None:
                continue
            xj, yj = p
            rj = radii[j]
            d = math.hypot(x - xj, y - yj)
            if d + _EPS < float(r) + rj:
                return False
        return True

    # ==================================================================
    # (3) Construction / improvement
    # ==================================================================
    def _empty_placements() -> list:
        return [None] * n

    def to_coords(placements) -> list:
        """Convert internal placements list to CO-Bench `coords` list:
        None -> (-1, -1). Length-preserving; safe to drop straight into
        the solution dict as {'coords': to_coords(placements)}."""
        out = []
        for p in placements:
            if p is None:
                out.append((-1, -1))
            else:
                out.append((float(p[0]), float(p[1])))
        return out

    def _candidate_grid(step: float) -> list:
        """Generate candidate (x, y) centers on a regular grid inside the
        container's bounding square, clipped to points within radius R."""
        pts = []
        x = cx - R
        while x <= cx + R + 1e-12:
            y = cy - R
            while y <= cy + R + 1e-12:
                if math.hypot(x - cx, y - cy) <= R:
                    pts.append((x, y))
                y += step
            x += step
        return pts

    def prefix_grid_construct(grid_steps: int = 40) -> list:
        """Place circles 0, 1, 2, ... (smallest radii first, respecting the
        PREFIX constraint) at the first feasible grid point. Returns a
        placements list; placements[i] is None for any circle that could
        not be placed (and thus all circles with index >= that one are
        also None, preserving the prefix).

        `grid_steps` controls the candidate grid density inside the
        bounding square; the grid step is 2*R / grid_steps. Larger values
        find more positions at quadratic cost.
        """
        steps = max(4, int(grid_steps))
        step = (2.0 * R) / steps
        grid = _candidate_grid(step)
        # always include the container center as a strong first candidate
        grid.insert(0, (cx, cy))

        placements = _empty_placements()
        for i in range(n):
            r = radii[i]
            placed = False
            for (gx, gy) in grid:
                if space_for_circle(placements, r, gx, gy):
                    placements[i] = (gx, gy)
                    placed = True
                    break
            if not placed:
                # PREFIX property: stop -- can't pack circle i, so skip the rest.
                break
        return placements

    def front_packing_construct(num_angles: int = 36) -> list:
        """Heuristic "front" packing for prefix order. The first circle is
        placed at the container center; each subsequent circle is placed
        tangent to (a) the container wall, or (b) an already-placed circle,
        sampling `num_angles` orientations around each support. We take the
        first feasible candidate -- O(n * (n + num_angles)) per circle.

        Respects the PREFIX constraint: stops at the first circle that
        can't be placed and leaves all later circles as None.
        """
        placements = _empty_placements()
        if n == 0:
            return placements

        # circle 0: at the container center if possible, else at (cx + dx, cy)
        r0 = radii[0]
        if is_inside_container(cx, cy, r0):
            placements[0] = (cx, cy)
        else:
            # very degenerate -- shouldn't happen but be safe
            placements[0] = (cx, cy)

        angles = [2.0 * math.pi * k / max(1, int(num_angles))
                  for k in range(max(1, int(num_angles)))]

        for i in range(1, n):
            r = radii[i]
            cand = []
            # (a) tangent to container wall, sampled around the container
            for a in angles:
                d_center = R - r
                x = cx + d_center * math.cos(a)
                y = cy + d_center * math.sin(a)
                cand.append((x, y))
            # (b) tangent to each placed circle, sampled around that circle
            for j in range(i):
                if placements[j] is None:
                    continue
                xj, yj = placements[j]
                d = radii[j] + r
                for a in angles:
                    cand.append((xj + d * math.cos(a),
                                 yj + d * math.sin(a)))

            placed = False
            for (x, y) in cand:
                if space_for_circle(placements, r, x, y):
                    placements[i] = (x, y)
                    placed = True
                    break
            if not placed:
                break  # prefix property: stop here
        return placements

    def try_place_next(placements, num_angles: int = 72,
                       grid_steps: int = 30) -> Optional[list]:
        """Try to extend `placements` by ONE more circle (the lowest-index
        unplaced circle, which is required by the prefix constraint).
        Combines wall-tangent / pair-tangent candidates with a fine grid
        sweep. Returns a NEW placements list with that one circle added
        if successful, else None.
        """
        # find the next index to place (must be the smallest None index)
        next_i = None
        for i in range(n):
            if placements[i] is None:
                next_i = i
                break
        if next_i is None:
            return None  # all already placed
        # sanity: prefix property must hold in the input
        for j in range(next_i):
            if placements[j] is None:
                return None  # caller broke the prefix

        r = radii[next_i]
        cand = []
        # angular candidates around walls + each placed circle
        ang = [2.0 * math.pi * k / max(1, int(num_angles))
               for k in range(max(1, int(num_angles)))]
        d_center = R - r
        for a in ang:
            cand.append((cx + d_center * math.cos(a),
                         cy + d_center * math.sin(a)))
        for j in range(next_i):
            xj, yj = placements[j]
            d = radii[j] + r
            for a in ang:
                cand.append((xj + d * math.cos(a),
                             yj + d * math.sin(a)))
        # grid candidates as fallback
        step = (2.0 * R) / max(4, int(grid_steps))
        cand.extend(_candidate_grid(step))

        for (x, y) in cand:
            if space_for_circle(placements, r, x, y):
                new_pl = list(placements)
                new_pl[next_i] = (x, y)
                return new_pl
        return None

    def apply_local_shift(placements, t_limit_s: float = 2.0,
                          delta: Optional[float] = None,
                          seed: Optional[int] = None) -> list:
        """Local-search shifter: repeatedly pick a placed circle, propose a
        small random displacement, accept if it remains feasible. The goal
        is to compact the layout so a subsequent call to `try_place_next`
        succeeds. Returns a (possibly new) placements list; the caller
        should re-run `try_place_next` after this.

        `delta` is the per-step displacement magnitude (default 0.1 * R).
        Stops after t_limit_s seconds or when 5*n consecutive proposals
        fail in a row, whichever first.
        """
        rng = random.Random(seed)
        pl = [None if p is None else (float(p[0]), float(p[1]))
              for p in placements]
        placed_idx = [i for i, p in enumerate(pl) if p is not None]
        if not placed_idx:
            return pl
        d = float(delta) if delta is not None else 0.1 * R
        t0 = time.time()
        safety = 0.05
        fail_streak = 0
        max_fail = max(20, 5 * len(placed_idx))
        while (time.time() - t0) < t_limit_s - safety:
            if fail_streak >= max_fail:
                # cool down: try smaller step instead of giving up entirely
                if d < 1e-4 * R:
                    break
                d *= 0.5
                fail_streak = 0
            i = rng.choice(placed_idx)
            xi, yi = pl[i]
            theta = rng.random() * 2.0 * math.pi
            step = d * (0.25 + 0.75 * rng.random())  # vary magnitude
            nx = xi + step * math.cos(theta)
            ny = yi + step * math.sin(theta)
            # remove circle i, test, restore
            saved = pl[i]
            pl[i] = None
            if space_for_circle(pl, radii[i], nx, ny):
                pl[i] = (nx, ny)
                fail_streak = 0
            else:
                pl[i] = saved
                fail_streak += 1
        return pl

    return {
        # (1) queries
        "n_circles": n_circles,
        "circle_radius": circle_radius,
        "container_dims": container_dims,
        # (2) feasibility primitives
        "is_inside_container": is_inside_container,
        "circles_overlap": circles_overlap,
        "space_for_circle": space_for_circle,
        # (3) construction / improvement
        "to_coords": to_coords,
        "prefix_grid_construct": prefix_grid_construct,
        "front_packing_construct": front_packing_construct,
        "try_place_next": try_place_next,
        "apply_local_shift": apply_local_shift,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "n_circles",
        "input": "(no args)",
        "output": "int",
        "purpose": "Total number of circles n in the instance. O(1).",
    },
    {
        "name": "circle_radius",
        "input": "i: int  (0-based)",
        "output": "float",
        "purpose": (
            "Radius of circle `i`. Indices are 0-based and circles are "
            "sorted by radius in NON-DECREASING order, so circle 0 is the "
            "smallest. O(1)."
        ),
    },
    {
        "name": "container_dims",
        "input": "(no args)",
        "output": "dict  {'shape': 'circle', 'cx', 'cy', 'R'}",
        "purpose": (
            "Container geometry. For this task it is always a CIRCLE with "
            "center (cx, cy) and radius R."
        ),
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "is_inside_container",
        "input": "x: float, y: float, r: float",
        "output": "bool",
        "purpose": (
            "True iff circle (x, y, r) lies fully inside the container, "
            "i.e. dist((x,y), (cx,cy)) + r <= R within tolerance. O(1)."
        ),
    },
    {
        "name": "circles_overlap",
        "input": "c1_xyr: tuple[float,float,float], c2_xyr: tuple[float,float,float]",
        "output": "bool",
        "purpose": (
            "True iff the two circles (x,y,r) overlap (interiors intersect). "
            "Tangent contact is treated as non-overlapping. O(1). Use as a "
            "primitive for ad-hoc feasibility checks."
        ),
    },
    {
        "name": "space_for_circle",
        "input": ("placements: list[(x,y) | None] of length n, "
                  "r: float, x: float, y: float"),
        "output": "bool",
        "purpose": (
            "True iff a circle of radius `r` can be placed at (x, y) without "
            "leaving the container or overlapping any circle already in "
            "`placements`. `placements[i]` is the center of circle i or None "
            "if not yet placed. O(n)."
        ),
    },
    # ----- (3) Construction / improvement -----
    {
        "name": "to_coords",
        "input": "placements: list[(x,y) | None]",
        "output": "list[(float, float)]",
        "purpose": (
            "Convert an internal placements list to CO-Bench's `coords` "
            "list, replacing None entries with (-1, -1). Drop straight into "
            "the solution: {'coords': to_coords(placements)}. The PREFIX "
            "constraint requires that None entries form a suffix -- if any "
            "later circle is placed but an earlier one is None, eval_func "
            "will reject the solution."
        ),
    },
    {
        "name": "prefix_grid_construct",
        "input": "grid_steps: int = 40",
        "output": "list[(x,y) | None]  (length n)",
        "purpose": (
            "Greedy prefix construction: iterate circles in index order "
            "(smallest radius first, as required by the PREFIX constraint) "
            "and place each at the first feasible point on a regular grid "
            "of step 2R/grid_steps inside the container. Stops at the first "
            "circle that doesn't fit and leaves all later circles None. "
            "Excellent default warm start. O(n * grid_steps^2 * n)."
        ),
    },
    {
        "name": "front_packing_construct",
        "input": "num_angles: int = 36",
        "output": "list[(x,y) | None]  (length n)",
        "purpose": (
            "Front packing in prefix order: circle 0 goes at the container "
            "center; each later circle is placed at the first feasible "
            "point among (a) wall-tangent positions sampled at num_angles "
            "evenly-spaced angles around the container and (b) pair-tangent "
            "positions sampled around each already-placed circle. Often "
            "denser than `prefix_grid_construct`. Stops at the first failure "
            "to preserve the PREFIX constraint. O(n^2 * num_angles)."
        ),
    },
    {
        "name": "try_place_next",
        "input": ("placements: list[(x,y) | None], num_angles: int = 72, "
                  "grid_steps: int = 30"),
        "output": "list[(x,y) | None] | None",
        "purpose": (
            "Try to extend `placements` by placing ONE more circle (the "
            "lowest-index unplaced one, as required by the PREFIX rule). "
            "Tests wall-tangent, pair-tangent, and grid candidates. Returns "
            "a NEW placements list on success, or None if no feasible spot "
            "exists. Useful after `apply_local_shift` compacts an existing "
            "layout."
        ),
    },
    {
        "name": "apply_local_shift",
        "input": ("placements: list[(x,y) | None], t_limit_s: float = 2.0, "
                  "delta: float | None = None, seed: int | None = None"),
        "output": "list[(x,y) | None]",
        "purpose": (
            "Random-perturbation compactor: repeatedly nudge a random "
            "placed circle by up to `delta` (default 0.1 * R), accepting "
            "any feasible move. Goal is to free space so a subsequent call "
            "to `try_place_next` can fit one more circle. Halves `delta` "
            "when proposals stall; returns when t_limit_s elapses. Does "
            "NOT change which circles are placed -- combine with "
            "`try_place_next` for that."
        ),
    },
]
