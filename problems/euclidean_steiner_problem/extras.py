"""Per-problem extras for CO-Bench Euclidean Steiner Problem.

Provides primitive building blocks so the LLM can compose construction +
local-search heuristics without reinventing MST, Fermat-point geometry, etc.

Instance schema (from CO-Bench load_data):
  - points: list[(x, y)] terminal coordinates.

Solution schema:
  - {"steiner_points": list[(x, y)]} -- additional points; eval recomputes
    MST on the union of terminals + steiner_points. Feasible only when that
    MST length is <= MST on terminals alone. Lower MST_union / MST_terminals
    is better.

Tool groups:
  (1) Queries:        n_terminals, terminal_coord, distance
  (2) MST primitives: mst_length, mst_edges, mst_terminals_only_length
  (3) Construction:   fermat_point_3, add_fermat_points_for_mst_triples
  (4) Improvement:    local_relocate_steiner

All are optional. LLM may use any subset.
"""
from __future__ import annotations
import math
import random
import time
from typing import Iterable, Optional

import numpy as np


def extra_tools(instance: dict) -> dict:
    """Factory: returns ESTP-specific tool callables given the loaded instance."""
    terminals: list = list(instance["points"])
    n = len(terminals)
    T = np.asarray(terminals, dtype=np.float64) if n > 0 else np.zeros((0, 2))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _euclid(a, b) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _prim_mst(points: list) -> tuple[float, list[tuple[int, int]]]:
        """Return (total_length, edges as list of (i,j) index pairs in `points`)."""
        m = len(points)
        if m <= 1:
            return 0.0, []
        in_mst = [False] * m
        min_dist = [float("inf")] * m
        parent = [-1] * m
        min_dist[0] = 0.0
        total = 0.0
        edges: list[tuple[int, int]] = []
        for _ in range(m):
            u = -1
            best = float("inf")
            for j in range(m):
                if not in_mst[j] and min_dist[j] < best:
                    best = min_dist[j]
                    u = j
            if u == -1:
                break
            in_mst[u] = True
            if parent[u] != -1:
                edges.append((parent[u], u))
                total += best
            for v in range(m):
                if not in_mst[v]:
                    d = _euclid(points[u], points[v])
                    if d < min_dist[v]:
                        min_dist[v] = d
                        parent[v] = u
        return total, edges

    # Cache MST on terminals only (used by many tools; never changes).
    _mst_terms_length, _mst_terms_edges = _prim_mst(terminals)

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def n_terminals() -> int:
        """Number of original terminal points in the instance."""
        return int(n)

    def terminal_coord(i: int) -> tuple[float, float]:
        """Coordinates (x, y) of terminal i (0 <= i < n_terminals())."""
        if not (0 <= int(i) < n):
            raise ValueError(f"terminal index {i} out of range [0, {n})")
        x, y = terminals[int(i)]
        return (float(x), float(y))

    def distance(p1, p2) -> float:
        """Euclidean distance between two 2D points (each as (x, y) tuple)."""
        return float(_euclid(p1, p2))

    # ==================================================================
    # (2) MST primitives
    # ==================================================================
    def mst_length(extra_points: Optional[Iterable] = None) -> float:
        """Total Euclidean length of the MST built over the terminals UNION the
        given `extra_points` (Steiner candidates). With extra_points=None or
        empty, returns the MST length over terminals only.

        This mirrors how `eval_func` scores a solution -- use it to evaluate
        candidate Steiner sets without round-tripping through the framework."""
        extras = list(extra_points) if extra_points else []
        if not extras:
            return float(_mst_terms_length)
        pts = terminals + [(float(p[0]), float(p[1])) for p in extras]
        total, _ = _prim_mst(pts)
        return float(total)

    def mst_edges(extra_points: Optional[Iterable] = None) -> list[tuple[int, int]]:
        """MST edges over (terminals + extra_points), returned as list of
        (i, j) index pairs. Indices [0, n_terminals()) refer to terminals;
        indices >= n_terminals() refer to extra_points in the order given."""
        extras = list(extra_points) if extra_points else []
        pts = terminals + [(float(p[0]), float(p[1])) for p in extras]
        _, edges = _prim_mst(pts)
        return edges

    def mst_terminals_only_length() -> float:
        """MST length on terminals only -- the BASELINE / upper bound any
        feasible Steiner solution must NOT exceed (eval_func raises if violated).
        Cached, so this is O(1) on repeated calls."""
        return float(_mst_terms_length)

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def fermat_point_3(a, b, c) -> tuple[float, float]:
        """Geometric median (a.k.a. Fermat / Torricelli point) of three 2D
        points -- the point that minimizes the sum of Euclidean distances to
        a, b, c. For a triangle whose every angle is < 120 degrees, this point
        lies strictly inside (a true Steiner point); if one angle >= 120 deg,
        the optimum coincides with that obtuse-angle vertex.

        Returns (x, y). Uses the closed-form 120-deg test plus Weiszfeld
        iteration (5 steps is usually enough)."""
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        cx, cy = float(c[0]), float(c[1])

        # Detect a vertex with angle >= 120 degrees: cos(angle) <= -1/2.
        def _angle_cos(v1, v2):
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            if n1 == 0.0 or n2 == 0.0:
                return 1.0
            return (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)

        cos_a = _angle_cos((bx - ax, by - ay), (cx - ax, cy - ay))
        cos_b = _angle_cos((ax - bx, ay - by), (cx - bx, cy - by))
        cos_c = _angle_cos((ax - cx, ay - cy), (bx - cx, by - cy))
        if cos_a <= -0.5:
            return (ax, ay)
        if cos_b <= -0.5:
            return (bx, by)
        if cos_c <= -0.5:
            return (cx, cy)

        # Weiszfeld iteration toward the geometric median.
        px = (ax + bx + cx) / 3.0
        py = (ay + by + cy) / 3.0
        for _ in range(64):
            d1 = math.hypot(px - ax, py - ay) + 1e-12
            d2 = math.hypot(px - bx, py - by) + 1e-12
            d3 = math.hypot(px - cx, py - cy) + 1e-12
            w = 1.0 / d1 + 1.0 / d2 + 1.0 / d3
            nx = (ax / d1 + bx / d2 + cx / d3) / w
            ny = (ay / d1 + by / d2 + cy / d3) / w
            if abs(nx - px) + abs(ny - py) < 1e-9:
                px, py = nx, ny
                break
            px, py = nx, ny
        return (float(px), float(py))

    def add_fermat_points_for_mst_triples(min_improvement: float = 1e-9) -> list[tuple[float, float]]:
        """Greedy Steiner-point seeding driven by the terminal MST.

        Strategy: walk the terminal MST; every interior terminal v that has at
        least two MST neighbors becomes the apex of a triple (u, v, w) where
        u, w are its two NEAREST MST neighbors (in terminal coordinates). For
        each such triple, compute its Fermat point f. If inserting f reduces
        the local 2-edge sum |uv| + |vw| (i.e., |uf|+|vf|+|wf| < |uv|+|vw|),
        accept f. Only candidates that strictly improve are returned.

        Returns a list of (x, y) Steiner-point candidates; safe to pass straight
        as the `steiner_points` solution but you should run `mst_length([...])`
        on it (feasibility = MST_union <= MST_terminals) before submitting."""
        # Build terminal adjacency from cached MST edges.
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for i, j in _mst_terms_edges:
            adj[i].append(j)
            adj[j].append(i)

        accepted: list[tuple[float, float]] = []
        for v in range(n):
            nbrs = adj[v]
            if len(nbrs) < 2:
                continue
            # pick the two nearest neighbors of v (in terminal coords)
            nbrs_sorted = sorted(nbrs, key=lambda j: _euclid(terminals[v], terminals[j]))
            u, w = nbrs_sorted[0], nbrs_sorted[1]
            f = fermat_point_3(terminals[u], terminals[v], terminals[w])
            d_uv = _euclid(terminals[u], terminals[v])
            d_vw = _euclid(terminals[v], terminals[w])
            d_uf = _euclid(terminals[u], f)
            d_vf = _euclid(terminals[v], f)
            d_wf = _euclid(terminals[w], f)
            if (d_uf + d_vf + d_wf) + min_improvement < (d_uv + d_vw):
                accepted.append(f)
        return accepted

    # ==================================================================
    # (4) Improvement
    # ==================================================================
    def local_relocate_steiner(
        steiner_points: Iterable,
        time_limit_s: float = 2.0,
        step: float = 0.05,
    ) -> list[tuple[float, float]]:
        """Coordinate-descent refinement of existing Steiner points.

        Iteratively perturbs each Steiner point by +/- `step` (in instance-
        coordinate units, scaled by the bounding-box diagonal) along x and y.
        Keeps the change iff the resulting MST length (terminals + all current
        Steiner points) strictly decreases. Shrinks `step` by 0.5x once no
        axis-aligned move helps. Stops at time_limit_s or when step < 1e-6.

        Returns a refined list of (x, y) Steiner points (NEVER drops a point;
        if a point is useless the MST simply won't connect through it, but it
        is still safe -- only check feasibility before submitting)."""
        sp = [(float(p[0]), float(p[1])) for p in steiner_points]
        if not sp:
            return sp

        # scale step by bbox diagonal so it's instance-relative
        if n > 0:
            xs = [p[0] for p in terminals]
            ys = [p[1] for p in terminals]
            diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
        else:
            diag = 1.0
        s = float(step) * diag

        def _len(extras):
            return _prim_mst(terminals + extras)[0]

        cur_len = _len(sp)
        t0 = time.time()
        safety = 0.05
        while s > 1e-6 * diag and (time.time() - t0) < time_limit_s - safety:
            improved_any = False
            for i in range(len(sp)):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                x, y = sp[i]
                best = (x, y)
                best_len = cur_len
                for dx, dy in ((s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)):
                    sp[i] = (x + dx, y + dy)
                    L = _len(sp)
                    if L + 1e-12 < best_len:
                        best_len = L
                        best = (x + dx, y + dy)
                sp[i] = best
                if best_len + 1e-12 < cur_len:
                    cur_len = best_len
                    improved_any = True
            if not improved_any:
                s *= 0.5
        return sp

    # ==================================================================
    # (5) Solution-dict builder + one-shot solver
    # ==================================================================
    def make_solution(steiner_points: Optional[Iterable] = None) -> dict:
        """Wrap a list of (x, y) Steiner points into the EXACT dict shape
        eval_func expects: {'steiner_points': list[(x, y)]}. Automatically
        filters out any point that would make the MST_union exceed
        MST_terminals (which would make the solution infeasible) -- if even
        the empty set's MST equals the terminal MST, returns an empty list,
        which is always feasible."""
        pts = list(steiner_points) if steiner_points else []
        # Feasibility guard: drop any point whose inclusion makes
        # MST_union > MST_terminals.
        if pts:
            cur = mst_length(pts)
            if cur > _mst_terms_length + 1e-9:
                # try dropping points one-by-one greedily until feasible
                kept: list = []
                for p in pts:
                    trial = kept + [p]
                    if mst_length(trial) <= _mst_terms_length + 1e-9:
                        kept = trial
                pts = kept
        return {"steiner_points": [(float(p[0]), float(p[1])) for p in pts]}

    def solve_default(time_limit_s: float = 5.0) -> dict:
        """ONE-SHOT STRONG SOLVER. Returns the complete solution dict
        {'steiner_points': list[(x, y)]} ready to return directly.

        Strategy: seed Steiner candidates with add_fermat_points_for_mst_
        triples (which only returns points that locally improve), then
        polish with local_relocate_steiner under the remaining time
        budget. Always returns a feasible solution (empty list at worst,
        which trivially satisfies MST_union <= MST_terminals).

        Use as the FIRST thing your solve() function calls. ONE LINE:
            return tools['solve_default'](time_limit_s=5)
        """
        if n < 3:
            return {"steiner_points": []}
        seeds = add_fermat_points_for_mst_triples()
        if not seeds:
            return {"steiner_points": []}
        polished = local_relocate_steiner(
            seeds, time_limit_s=max(0.5, time_limit_s - 0.1)
        )
        return make_solution(polished)

    return {
        # one-shot + builder (CALL FIRST)
        "solve_default": solve_default,
        "make_solution": make_solution,
        # construction / improvement
        "add_fermat_points_for_mst_triples": add_fermat_points_for_mst_triples,
        "local_relocate_steiner": local_relocate_steiner,
        "fermat_point_3": fermat_point_3,
        # mst primitives
        "mst_length": mst_length,
        "mst_edges": mst_edges,
        "mst_terminals_only_length": mst_terminals_only_length,
        # queries
        "n_terminals": n_terminals,
        "terminal_coord": terminal_coord,
        "distance": distance,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- ONE-SHOT STRONG SOLVER (call this first!) -----
    {
        "name": "solve_default",
        "input": "time_limit_s: float = 5.0",
        "output": "dict {'steiner_points': list[(x, y)]}",
        "purpose": (
            "RECOMMENDED START: returns a complete solution dict ready to return "
            "directly. Seeds Steiner candidates via add_fermat_points_for_mst_"
            "triples (only locally improving points), then polishes with "
            "local_relocate_steiner under the remaining budget. Always feasible "
            "(empty list at worst). ONE LINE: "
            "`return tools['solve_default'](time_limit_s=5)`."
        ),
    },
    {
        "name": "make_solution",
        "input": "steiner_points: Iterable[(x, y)] | None = None",
        "output": "dict {'steiner_points': list[(x, y)]}",
        "purpose": (
            "Build the EXACT solution dict shape eval_func wants from a list of "
            "(x, y) Steiner points. AUTOMATICALLY drops any point that would "
            "make MST_union exceed MST_terminals -- so the returned dict is "
            "always feasible. Use on the output of add_fermat_points_for_mst_"
            "triples() / local_relocate_steiner() so you never return the "
            "wrong dict shape."
        ),
    },
    # ----- Queries -----
    {
        "name": "n_terminals",
        "input": "(no args)",
        "output": "int",
        "purpose": (
            "Number of original terminal points in the instance "
            "(== len(instance['points']))."
        ),
    },
    {
        "name": "terminal_coord",
        "input": "i: int",
        "output": "tuple[float, float]",
        "purpose": (
            "Coordinates (x, y) of terminal i. Raises ValueError if i is out "
            "of range [0, n_terminals())."
        ),
    },
    {
        "name": "distance",
        "input": "p1: (x, y), p2: (x, y)",
        "output": "float",
        "purpose": "Euclidean distance between any two 2D points.",
    },
    # ----- MST primitives -----
    {
        "name": "mst_length",
        "input": "extra_points: Iterable[(x, y)] | None = None",
        "output": "float",
        "purpose": (
            "Total Euclidean length of the MST built on (terminals + "
            "extra_points). Pass extra_points=None or [] to get the baseline "
            "MST length on terminals only. This is exactly what eval_func "
            "compares against, so use it to score candidate steiner_points "
            "lists without going through tools['objective']."
        ),
    },
    {
        "name": "mst_edges",
        "input": "extra_points: Iterable[(x, y)] | None = None",
        "output": "list[tuple[int, int]]",
        "purpose": (
            "Edges of the MST on (terminals + extra_points) as (i, j) index "
            "pairs. Indices < n_terminals() refer to terminals; indices >= "
            "n_terminals() refer to extra_points in the order passed. Useful "
            "for finding 3-terminal triples to insert Fermat points into."
        ),
    },
    {
        "name": "mst_terminals_only_length",
        "input": "(no args)",
        "output": "float",
        "purpose": (
            "MST length on terminals only -- the FEASIBILITY UPPER BOUND a "
            "solution must NOT exceed (eval_func raises if it does). Cached, "
            "O(1) on repeat calls."
        ),
    },
    # ----- Construction -----
    {
        "name": "fermat_point_3",
        "input": "a: (x, y), b: (x, y), c: (x, y)",
        "output": "tuple[float, float]",
        "purpose": (
            "Fermat / Torricelli point of three 2D points: the geometric "
            "median that minimizes the sum of Euclidean distances to a, b, c. "
            "For a triangle with all angles < 120 deg, this is a true interior "
            "Steiner point that beats any two-edge MST connecting the three. "
            "If one angle >= 120 deg, returns that obtuse-angle vertex itself "
            "(no Steiner gain available)."
        ),
    },
    {
        "name": "add_fermat_points_for_mst_triples",
        "input": "min_improvement: float = 1e-9",
        "output": "list[tuple[float, float]]",
        "purpose": (
            "Greedy Steiner seeding: walks the terminal MST and, for every "
            "interior terminal v with >=2 MST neighbors, takes the two NEAREST "
            "MST neighbors (u, w) and computes the Fermat point of (u, v, w). "
            "Returns only the Fermat points that strictly improve the local "
            "2-edge sum |uv|+|vw|. Solid warm-start steiner_points list; pair "
            "with local_relocate_steiner to polish."
        ),
    },
    # ----- Improvement -----
    {
        "name": "local_relocate_steiner",
        "input": "steiner_points: Iterable[(x, y)], time_limit_s: float = 2.0, step: float = 0.05",
        "output": "list[tuple[float, float]]",
        "purpose": (
            "Coordinate-descent refinement: perturbs each Steiner point along "
            "+/- x and +/- y by `step * bbox_diagonal` and keeps the move iff "
            "the MST length (over terminals + all Steiner points) strictly "
            "decreases. Halves step when no axis-aligned move helps. Cheap "
            "post-processing for any Steiner-point list. Does not drop points; "
            "verify feasibility (mst_length(result) <= mst_terminals_only_length()) "
            "before submitting."
        ),
    },
]
