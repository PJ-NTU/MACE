"""Per-problem extras for CO-Bench Packing Unequal Circles (Maximize AREA).

The task: a circular container of radius R centered at (cx, cy) and n circles
with given radii. Decide WHICH circles to pack and place each packed circle
inside the container without overlapping any other packed circle. Objective:
maximize the total AREA (sum of pi * r_i^2) of packed circles.

Solution schema (from CO-Bench):
    {"coords": [(x_i, y_i) for i in 0..n-1]}
A circle i is considered NOT packed iff its (x_i, y_i) equals (-1, -1)
exactly (within a small tolerance). All other entries must satisfy:
  - containment: distance((x_i, y_i), (cx, cy)) + r_i <= R
  - non-overlap: distance((x_i, y_i), (x_j, y_j)) >= r_i + r_j  for every
                 packed pair i != j

These extras provide primitives so the LLM can compose construction +
local-search heuristics without re-deriving geometry from scratch.

Tool groups:
  (1) Queries:        num_circles, container, radius_of,
                      max_possible_area, lower_bound_largest_area
  (2) Inspection:     placed_indices, is_placed, total_area,
                      container_clearance, pair_clearance,
                      circle_violations, is_feasible_solution
  (3) Construction:   unpacked_template, place_circle, unplace_circle,
                      greedy_by_area_first, greedy_pack_in_order,
                      random_feasible_position
  (4) Local search:   try_relocate_circle, try_add_circle, try_swap_in_out

All construction / local-search tools are IMMUTABLE: they return a new
coords list and do not mutate inputs. Container/coordinates are 0-indexed
to match CO-Bench's coords list.
"""
from __future__ import annotations

import math
import random
import time
from typing import Iterable, Optional


# Match the tolerance used by CO-Bench's eval_func exactly.
_TOL = 1e-5
# Geometry tolerance for our own checks (a touch tighter than _TOL so any
# placement we deem feasible will also pass CO-Bench's check).
_GEO_EPS = 1e-7


def extra_tools(instance: dict) -> dict:
    """Factory: returns Packing-Unequal-Circles-Area tool callables.

    Instance schema (from CO-Bench load_data):
      - n     : int
      - cx    : float
      - cy    : float
      - R     : float       (container radius)
      - radii : list[float] (one radius per circle, 0-indexed)
    """
    n = int(instance["n"])
    cx = float(instance["cx"])
    cy = float(instance["cy"])
    R = float(instance["R"])
    radii = [float(r) for r in instance["radii"]]
    if len(radii) != n:
        n = len(radii)

    # Pre-sort indices by radius desc -- biggest circles first (most "valuable"
    # per the area objective).
    order_area_desc = sorted(range(n), key=lambda i: -radii[i])

    # Trivial upper bound: sum of every circle's area (rarely achievable).
    _total_possible_area = sum(math.pi * r * r for r in radii)
    # Trivial lower bound: the largest single circle that fits the container.
    _lb_largest_area = max((math.pi * r * r for r in radii if r <= R), default=0.0)

    # ==================================================================
    # Local geometry helpers (closure-private)
    # ==================================================================
    def _is_unpacked_point(p) -> bool:
        # Mirror CO-Bench's "(-1, -1) within tol" convention.
        try:
            x, y = p
        except Exception:
            return False
        return abs(float(x) + 1.0) <= _TOL and abs(float(y) + 1.0) <= _TOL

    def _packed_idx(coords) -> list[int]:
        out = []
        for i in range(n):
            if i >= len(coords):
                break
            if not _is_unpacked_point(coords[i]):
                out.append(i)
        return out

    def _check_radius(i: int) -> None:
        if not (0 <= int(i) < n):
            raise ValueError(f"circle index {i} out of range [0, {n})")

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def num_circles() -> int:
        """Total number of circles n in the instance. O(1)."""
        return n

    def container() -> tuple[float, float, float]:
        """Container as (cx, cy, R). O(1)."""
        return (cx, cy, R)

    def radius_of(i: int) -> float:
        """Radius of circle `i` (0-based). O(1)."""
        _check_radius(i)
        return radii[i]

    def max_possible_area() -> float:
        """Upper bound on the total area objective: sum of pi*r_i^2 for ALL
        circles, regardless of whether they fit together. The true optimum is
        <= this value. Useful for gap estimation."""
        return _total_possible_area

    def lower_bound_largest_area() -> float:
        """Trivial lower bound: the area of the single LARGEST circle that
        fits inside the container alone (r <= R). Always achievable by placing
        that one circle at the container's center."""
        return _lb_largest_area

    # ==================================================================
    # (2) Inspection
    # ==================================================================
    def placed_indices(coords) -> list[int]:
        """Indices (0-based) of circles whose center is NOT (-1, -1). Uses the
        same tolerance rule as CO-Bench's eval_func."""
        return _packed_idx(coords)

    def is_placed(i: int, coords) -> bool:
        """True iff circle `i` has a real center in `coords` (not (-1, -1))."""
        _check_radius(i)
        if i >= len(coords):
            return False
        return not _is_unpacked_point(coords[i])

    def total_area(coords) -> float:
        """Sum of pi * r_i^2 over all PLACED circles in `coords`. This is the
        raw maximization objective the problem cares about (the CO-Bench
        framework wraps it as 1/raw via the spec's tools['objective'] for
        lower-better semantics, but here we return the raw area)."""
        s = 0.0
        for i in _packed_idx(coords):
            s += math.pi * radii[i] * radii[i]
        return s

    def container_clearance(i: int, coords) -> float:
        """Slack in the container constraint for circle `i`:
            R - (distance((x_i, y_i), (cx, cy)) + r_i)
        Non-negative => feasible (with tolerance _TOL). Returns +inf if the
        circle is unpacked (constraint trivially satisfied)."""
        _check_radius(i)
        if i >= len(coords) or _is_unpacked_point(coords[i]):
            return float("inf")
        x, y = coords[i]
        d = math.hypot(x - cx, y - cy)
        return R - (d + radii[i])

    def pair_clearance(i: int, j: int, coords) -> float:
        """Slack in the non-overlap constraint for the pair (i, j):
            distance(center_i, center_j) - (r_i + r_j)
        Non-negative => feasible. Returns +inf if either circle is unpacked."""
        _check_radius(i)
        _check_radius(j)
        if i == j:
            return float("inf")
        if (i >= len(coords) or j >= len(coords)
                or _is_unpacked_point(coords[i])
                or _is_unpacked_point(coords[j])):
            return float("inf")
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        d = math.hypot(x1 - x2, y1 - y2)
        return d - (radii[i] + radii[j])

    def circle_violations(coords) -> list[tuple]:
        """List of constraint violations in `coords`. Each entry is one of:
              ('container', i, slack)         -- circle i pokes outside (slack<0)
              ('overlap',   i, j, slack)      -- circles i and j overlap (slack<0)
              ('length',    msg)              -- coords has the wrong length
        Empty list => fully feasible (mirrors CO-Bench's eval_func rules
        with the same tolerance _TOL). Use to drive a repair loop."""
        v: list[tuple] = []
        if not isinstance(coords, list) or len(coords) != n:
            v.append(("length", f"len(coords)={len(coords) if hasattr(coords, '__len__') else '?'} != n={n}"))
            return v
        packed = _packed_idx(coords)
        for i in packed:
            x, y = coords[i]
            d = math.hypot(x - cx, y - cy)
            slack = R - (d + radii[i])
            if slack < -_TOL:
                v.append(("container", i, slack))
        for a, i in enumerate(packed):
            for j in packed[a + 1:]:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                d = math.hypot(x1 - x2, y1 - y2)
                slack = d - (radii[i] + radii[j])
                if slack < -_TOL:
                    v.append(("overlap", i, j, slack))
        return v

    def is_feasible_solution(coords) -> tuple[bool, Optional[str]]:
        """Local feasibility check that mirrors CO-Bench's eval_func without
        the framework round-trip. Returns (True, None) or (False, reason).
        Faster than tools['is_feasible'] for tight neighborhood-search loops.

        Note: a coords list where every entry is (-1, -1) is feasible (and
        scores 0), so the empty packing is always feasible."""
        if not isinstance(coords, list):
            return False, f"coords must be list, got {type(coords).__name__}"
        if len(coords) != n:
            return False, f"len(coords)={len(coords)} != n={n}"
        for v in circle_violations(coords):
            if v[0] == "container":
                _, i, slack = v
                return False, f"Circle {i} violates container by {-slack}"
            if v[0] == "overlap":
                _, i, j, slack = v
                return False, f"Circles {i} and {j} overlap by {-slack}"
            if v[0] == "length":
                return False, v[1]
        return True, None

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def unpacked_template() -> list[tuple]:
        """Return a fresh coords list of length n where every circle is marked
        unpacked: [(-1, -1), (-1, -1), ...]. Feasible (and scores 0). Use as
        the starting point for any construction heuristic."""
        return [(-1.0, -1.0) for _ in range(n)]

    def place_circle(coords, i: int, x: float, y: float) -> list[tuple]:
        """Return a NEW coords list with circle `i` placed at (x, y). Does NOT
        check feasibility -- caller should use is_feasible_solution / the
        clearance tools to validate. Use for hill-climbing and LNS."""
        _check_radius(i)
        new = list(coords)
        if len(new) < n:
            new = new + [(-1.0, -1.0)] * (n - len(new))
        elif len(new) > n:
            new = new[:n]
        new[i] = (float(x), float(y))
        return new

    def unplace_circle(coords, i: int) -> list[tuple]:
        """Return a NEW coords list with circle `i` marked unpacked
        ((-1, -1)). Use to ruin part of a solution before re-creating."""
        _check_radius(i)
        new = list(coords)
        if len(new) < n:
            new = new + [(-1.0, -1.0)] * (n - len(new))
        elif len(new) > n:
            new = new[:n]
        new[i] = (-1.0, -1.0)
        return new

    def _fits_at(coords, i: int, x: float, y: float) -> bool:
        """Internal: does circle i fit at (x, y) given currently placed?
        Uses _GEO_EPS so we stay strictly inside CO-Bench's _TOL band."""
        # container
        if math.hypot(x - cx, y - cy) + radii[i] > R + _GEO_EPS:
            return False
        for j in _packed_idx(coords):
            if j == i:
                continue
            xj, yj = coords[j]
            if math.hypot(x - xj, y - yj) < (radii[i] + radii[j]) - _GEO_EPS:
                return False
        return True

    def random_feasible_position(
        i: int,
        coords,
        attempts: int = 200,
        rng_seed: Optional[int] = None,
    ) -> Optional[tuple[float, float]]:
        """Sample up to `attempts` random points inside the disk of valid
        centers for circle `i` (a circle of radius R - r_i around (cx, cy))
        and return the first that does not overlap any circle already placed
        in `coords`. Returns (x, y) or None if no feasible sample was found.

        Does not modify `coords`. Use as a fallback when grid / corner
        placement fails."""
        _check_radius(i)
        if radii[i] > R:
            return None  # cannot fit at all
        rng = random.Random(rng_seed) if rng_seed is not None else random
        rmax = R - radii[i]
        for _ in range(int(attempts)):
            # uniform sample in disk of radius rmax around (cx, cy)
            rho = rmax * math.sqrt(rng.random())
            theta = 2.0 * math.pi * rng.random()
            x = cx + rho * math.cos(theta)
            y = cy + rho * math.sin(theta)
            if _fits_at(coords, i, x, y):
                return (x, y)
        return None

    def greedy_pack_in_order(
        order: Iterable[int],
        attempts_per_circle: int = 400,
        try_center_first: bool = True,
        rng_seed: Optional[int] = None,
    ) -> list[tuple]:
        """Sequential packing: walk `order` (a list of 0-based circle indices)
        and try to place each circle without violating the container or
        overlapping any previously-placed circle. For each circle:
          1. If `try_center_first` and nothing is placed yet, put the first
             circle at the container's center.
          2. Try a deterministic grid sweep over feasible (x, y) candidates
             constructed from already-placed circles and the container wall.
          3. If the grid fails, fall back to `attempts_per_circle` random
             samples (via random_feasible_position).
        Returns a coords list of length n with successfully-placed circles at
        their chosen centers and skipped circles at (-1, -1). The result is
        always feasible. O(n * attempts) in the worst case."""
        rng = random.Random(rng_seed) if rng_seed is not None else random
        coords = unpacked_template()

        # Candidate generator: for each new circle, try (a) the container
        # center, (b) the four cardinal extremes along the wall, (c) a hex
        # grid of points inside the valid disk, then (d) the contact points
        # implied by every (placed_circle, container_wall) and
        # (placed_circle, placed_circle) tangency.
        def _candidates(i: int):
            rmax = R - radii[i]
            if rmax < -_GEO_EPS:
                return []
            cand = []
            if try_center_first:
                cand.append((cx, cy))
            # axis-aligned wall points
            cand.extend([
                (cx + rmax, cy),
                (cx - rmax, cy),
                (cx, cy + rmax),
                (cx, cy - rmax),
            ])
            # tangencies: against each placed circle, two contact points on
            # the line joining their centers extended outward
            placed_now = _packed_idx(coords)
            for j in placed_now:
                xj, yj = coords[j]
                # the locus of feasible centers for i tangent to j is a circle
                # of radius (r_i + r_j) around (xj, yj); sample 8 evenly
                rij = radii[i] + radii[j]
                for k in range(8):
                    theta = 2.0 * math.pi * k / 8
                    cand.append((xj + rij * math.cos(theta),
                                 yj + rij * math.sin(theta)))
            # coarse grid as a fallback
            grid_n = 8
            for a in range(grid_n + 1):
                for b in range(grid_n + 1):
                    gx = cx - rmax + 2 * rmax * a / max(1, grid_n)
                    gy = cy - rmax + 2 * rmax * b / max(1, grid_n)
                    cand.append((gx, gy))
            return cand

        for i in order:
            if not (0 <= int(i) < n):
                continue
            i = int(i)
            if radii[i] > R + _GEO_EPS:
                continue  # cannot fit at all
            placed_here = False
            for (x, y) in _candidates(i):
                if _fits_at(coords, i, x, y):
                    coords[i] = (float(x), float(y))
                    placed_here = True
                    break
            if not placed_here:
                pos = random_feasible_position(
                    i, coords, attempts=attempts_per_circle,
                    rng_seed=rng.randint(0, 2**31 - 1),
                )
                if pos is not None:
                    coords[i] = pos
        return coords

    def greedy_by_area_first(
        attempts_per_circle: int = 400,
        rng_seed: Optional[int] = None,
    ) -> list[tuple]:
        """Convenience wrapper: greedy_pack_in_order with circles sorted by
        radius DESCENDING. Places the largest (most-valuable, per the area
        objective) circles first so that the harder-to-fit ones do not get
        boxed out by many small circles. Good default warm start."""
        return greedy_pack_in_order(
            list(order_area_desc),
            attempts_per_circle=attempts_per_circle,
            rng_seed=rng_seed,
        )

    # ==================================================================
    # (4) Local search
    # ==================================================================
    def try_relocate_circle(
        coords,
        i: int,
        attempts: int = 200,
        rng_seed: Optional[int] = None,
    ) -> Optional[list[tuple]]:
        """Try to move ALREADY-placed circle `i` to a new feasible position
        (sampled randomly inside its valid disk, ignoring its current
        position). Returns a NEW coords list if a feasible new spot was found,
        else None. Useful for escaping local optima when total_area cannot be
        improved by adding more circles directly."""
        _check_radius(i)
        if not is_placed(i, coords):
            return None
        # temporarily remove i so we don't self-overlap
        without_i = unplace_circle(coords, i)
        pos = random_feasible_position(
            i, without_i, attempts=attempts, rng_seed=rng_seed,
        )
        if pos is None:
            return None
        return place_circle(without_i, i, pos[0], pos[1])

    def try_add_circle(
        coords,
        i: int,
        attempts: int = 400,
        rng_seed: Optional[int] = None,
    ) -> Optional[list[tuple]]:
        """Try to add circle `i` (currently unpacked) to `coords` without
        violating any constraint. Returns a NEW coords list on success, else
        None. Increases total_area by pi * r_i^2 when it succeeds."""
        _check_radius(i)
        if is_placed(i, coords):
            return None
        pos = random_feasible_position(
            i, coords, attempts=attempts, rng_seed=rng_seed,
        )
        if pos is None:
            # also try the deterministic candidates once
            # (mirrors greedy_pack_in_order's strategy for a single circle)
            sweep = greedy_pack_in_order([i], attempts_per_circle=0)
            if is_placed(i, sweep):
                merged = list(coords)
                merged[i] = sweep[i]
                ok, _ = is_feasible_solution(merged)
                if ok:
                    return merged
            return None
        return place_circle(coords, i, pos[0], pos[1])

    def try_swap_in_out(
        coords,
        out_i: int,
        in_i: int,
        attempts: int = 400,
        rng_seed: Optional[int] = None,
    ) -> Optional[list[tuple]]:
        """Remove circle `out_i` (must currently be placed) and try to place
        circle `in_i` (must currently be unpacked) somewhere feasible. Returns
        a NEW coords list on success, else None. Only useful when r[in_i] >
        r[out_i] (otherwise the swap reduces total_area), but the tool does
        not enforce that -- caller decides when to call it."""
        _check_radius(out_i)
        _check_radius(in_i)
        if not is_placed(out_i, coords):
            return None
        if is_placed(in_i, coords):
            return None
        intermediate = unplace_circle(coords, out_i)
        pos = random_feasible_position(
            in_i, intermediate, attempts=attempts, rng_seed=rng_seed,
        )
        if pos is None:
            return None
        return place_circle(intermediate, in_i, pos[0], pos[1])

    return {
        # (1) queries
        "num_circles": num_circles,
        "container": container,
        "radius_of": radius_of,
        "max_possible_area": max_possible_area,
        "lower_bound_largest_area": lower_bound_largest_area,
        # (2) inspection
        "placed_indices": placed_indices,
        "is_placed": is_placed,
        "total_area": total_area,
        "container_clearance": container_clearance,
        "pair_clearance": pair_clearance,
        "circle_violations": circle_violations,
        "is_feasible_solution": is_feasible_solution,
        # (3) construction
        "unpacked_template": unpacked_template,
        "place_circle": place_circle,
        "unplace_circle": unplace_circle,
        "random_feasible_position": random_feasible_position,
        "greedy_pack_in_order": greedy_pack_in_order,
        "greedy_by_area_first": greedy_by_area_first,
        # (4) local search
        "try_relocate_circle": try_relocate_circle,
        "try_add_circle": try_add_circle,
        "try_swap_in_out": try_swap_in_out,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "num_circles",
        "input": "(no args)",
        "output": "int",
        "purpose": "Total number of circles n in the instance. O(1).",
    },
    {
        "name": "container",
        "input": "(no args)",
        "output": "(cx, cy, R)",
        "purpose": (
            "Container center coordinates and radius as a 3-tuple. Use to "
            "place candidate centers without re-reading instance keys."
        ),
    },
    {
        "name": "radius_of",
        "input": "i: int  (0-based circle index)",
        "output": "float",
        "purpose": "Radius of circle `i`. Circle indices are 0-based and match the coords list. O(1).",
    },
    {
        "name": "max_possible_area",
        "input": "(no args)",
        "output": "float",
        "purpose": (
            "Upper bound on the maximum-area objective: sum of pi*r_i^2 over "
            "ALL circles (regardless of geometric feasibility). The true "
            "optimum is <= this. Use to estimate optimality gap."
        ),
    },
    {
        "name": "lower_bound_largest_area",
        "input": "(no args)",
        "output": "float",
        "purpose": (
            "Trivial lower bound: pi * (largest r_i with r_i <= R)^2, achievable "
            "by packing just that one circle at the container's center."
        ),
    },
    # ----- (2) Inspection -----
    {
        "name": "placed_indices",
        "input": "coords: list[(float, float)]",
        "output": "list[int]",
        "purpose": (
            "Indices of circles whose center in `coords` is NOT (-1, -1). "
            "Uses the same tolerance as CO-Bench's eval_func."
        ),
    },
    {
        "name": "is_placed",
        "input": "i: int, coords: list[(float, float)]",
        "output": "bool",
        "purpose": "True iff circle `i` is currently packed in `coords`.",
    },
    {
        "name": "total_area",
        "input": "coords: list[(float, float)]",
        "output": "float",
        "purpose": (
            "Sum of pi * r_i^2 over all PLACED circles in `coords`. This is "
            "the raw maximization objective the problem wants you to make as "
            "LARGE as possible. (Note: tools['objective'] returns 1/raw "
            "because the framework standardizes on lower-better. Call "
            "total_area directly when you want the natural raw value.)"
        ),
    },
    {
        "name": "container_clearance",
        "input": "i: int, coords: list[(float, float)]",
        "output": "float",
        "purpose": (
            "Slack in the container constraint for circle `i`: "
            "R - (distance(center_i, (cx, cy)) + r_i). Non-negative <=> feasible. "
            "Returns +inf if `i` is unpacked."
        ),
    },
    {
        "name": "pair_clearance",
        "input": "i: int, j: int, coords: list[(float, float)]",
        "output": "float",
        "purpose": (
            "Slack in the non-overlap constraint for circles (i, j): "
            "distance(center_i, center_j) - (r_i + r_j). Non-negative <=> "
            "feasible. Returns +inf if either is unpacked."
        ),
    },
    {
        "name": "circle_violations",
        "input": "coords: list[(float, float)]",
        "output": "list[tuple]",
        "purpose": (
            "List of constraint violations in `coords`. Each entry is one of "
            "('container', i, slack), ('overlap', i, j, slack), or "
            "('length', msg). Empty list means fully feasible. Use to drive "
            "a repair loop."
        ),
    },
    {
        "name": "is_feasible_solution",
        "input": "coords: list[(float, float)]",
        "output": "(bool, str | None)",
        "purpose": (
            "Local feasibility check (mirrors CO-Bench's eval_func rules) "
            "without the framework round-trip. Faster than tools['is_feasible'] "
            "for tight neighborhood-search loops. NOTE: the all-unpacked "
            "list [(-1, -1)] * n is feasible (scores 0)."
        ),
    },
    # ----- (3) Construction -----
    {
        "name": "unpacked_template",
        "input": "(no args)",
        "output": "list[(float, float)]",
        "purpose": (
            "Fresh coords list of length n with every circle marked unpacked "
            "((-1, -1)). Feasible baseline; scores 0. Starting point for any "
            "construction heuristic."
        ),
    },
    {
        "name": "place_circle",
        "input": "coords: list[(float, float)], i: int, x: float, y: float",
        "output": "list[(float, float)]",
        "purpose": (
            "Return a NEW coords list with circle `i` set to (x, y). Does NOT "
            "check feasibility -- call is_feasible_solution or the clearance "
            "tools afterwards. Immutable: does not mutate the input."
        ),
    },
    {
        "name": "unplace_circle",
        "input": "coords: list[(float, float)], i: int",
        "output": "list[(float, float)]",
        "purpose": (
            "Return a NEW coords list with circle `i` marked unpacked "
            "((-1, -1)). Use to remove a circle before trying a better "
            "placement (ruin step of LNS)."
        ),
    },
    {
        "name": "random_feasible_position",
        "input": "i: int, coords: list[(float, float)], attempts: int = 200, rng_seed: int | None = None",
        "output": "(float, float) | None",
        "purpose": (
            "Sample up to `attempts` uniform-random centers inside the disk "
            "of valid positions for circle `i` (radius R - r_i around (cx, cy)) "
            "and return the first one that overlaps no circle currently in "
            "`coords`. Returns None if none of the samples is feasible. Does "
            "NOT modify `coords`."
        ),
    },
    {
        "name": "greedy_pack_in_order",
        "input": (
            "order: Iterable[int], attempts_per_circle: int = 400, "
            "try_center_first: bool = True, rng_seed: int | None = None"
        ),
        "output": "list[(float, float)]",
        "purpose": (
            "Sequential packing: walks `order` and tries to place each circle "
            "without violating any constraint. Tries a deterministic "
            "candidate sweep (container center, four wall points, tangency "
            "points against each placed circle, a coarse grid), then falls "
            "back to random sampling. Skipped circles stay (-1, -1). Always "
            "returns a feasible solution."
        ),
    },
    {
        "name": "greedy_by_area_first",
        "input": "attempts_per_circle: int = 400, rng_seed: int | None = None",
        "output": "list[(float, float)]",
        "purpose": (
            "Convenience: greedy_pack_in_order with circles sorted by radius "
            "DESCENDING. Places the largest (highest-area) circles first so "
            "easy small ones do not box them out. Excellent default warm "
            "start for the area objective. Always feasible."
        ),
    },
    # ----- (4) Local search -----
    {
        "name": "try_relocate_circle",
        "input": "coords: list[(float, float)], i: int, attempts: int = 200, rng_seed: int | None = None",
        "output": "list[(float, float)] | None",
        "purpose": (
            "Try to MOVE the already-placed circle `i` to a new random "
            "feasible spot (its prior position is freed first so it cannot "
            "block itself). Returns a NEW coords list on success, else None. "
            "Use to compact the layout so additional circles can be added."
        ),
    },
    {
        "name": "try_add_circle",
        "input": "coords: list[(float, float)], i: int, attempts: int = 400, rng_seed: int | None = None",
        "output": "list[(float, float)] | None",
        "purpose": (
            "Try to ADD currently-unpacked circle `i` to `coords` without "
            "violating any constraint. Returns a NEW coords list (with "
            "total_area increased by pi * r_i^2) on success, else None. The "
            "core 'improve' move for the area objective."
        ),
    },
    {
        "name": "try_swap_in_out",
        "input": (
            "coords: list[(float, float)], out_i: int, in_i: int, "
            "attempts: int = 400, rng_seed: int | None = None"
        ),
        "output": "list[(float, float)] | None",
        "purpose": (
            "Remove placed circle `out_i` and try to place currently-unpacked "
            "circle `in_i` somewhere feasible. Returns a NEW coords list on "
            "success, else None. Profitable when r[in_i] > r[out_i] (area "
            "gain pi*(r_in^2 - r_out^2) > 0). Caller decides when to invoke."
        ),
    },
]
