"""Per-problem extras for CO-Bench Graph Colouring.

Provides primitive building blocks so the LLM can compose construction +
local-search heuristics (greedy, DSATUR, color-reduction) and optionally
call an exact ILP solver. All vertices are 1-indexed (1..n) and colors
are positive integers (1..n), matching the CO-Bench solution dict shape:
    {vertex_id (int) -> color (int >= 1)}.

Tool groups:
  (1) Queries:        adjacency, degree, n_vertices, n_edges
  (2) Feasibility:    color_conflicts, is_proper_coloring, colors_used,
                      saturation
  (3) Construction:   greedy_color, dsatur_color, apply_recolor_vertex,
                      recolor_to_minimize_colors
  (4) Exact (heavy):  ilp_chromatic_number

All are optional. The LLM may use any subset or write everything from scratch.
"""
from __future__ import annotations
from typing import Optional, Iterable


def extra_tools(instance: dict) -> dict:
    """Factory: returns Graph-Colouring-specific tool callables for `instance`.

    Instance schema (from CO-Bench Graph Colouring load_data):
      - n:         int, number of vertices (vertices are 1..n).
      - edges:     list of (u, v) tuples.
      - adjacency: dict[int -> set[int]], 1-indexed adjacency list.
    """
    n: int = int(instance["n"])
    edges = instance["edges"]
    adj: dict[int, set[int]] = instance["adjacency"]

    # Normalize adjacency to tuple-of-sorted-ints once -- repeated iteration
    # is faster on lists than sets, and stable order makes greedy reproducible.
    adj_list: dict[int, list[int]] = {
        v: sorted(adj.get(v, set())) for v in range(1, n + 1)
    }
    deg: dict[int, int] = {v: len(adj_list[v]) for v in range(1, n + 1)}
    m_edges = sum(deg.values()) // 2  # each edge counted twice in adjacency

    # Brooks-style upper bound on colors needed.
    max_deg = max(deg.values()) if n > 0 else 0

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def adjacency(v: int) -> list:
        if not (1 <= int(v) <= n):
            raise ValueError(f"vertex={v} out of range [1, {n}]")
        return list(adj_list[int(v)])

    def degree(v: int) -> int:
        if not (1 <= int(v) <= n):
            raise ValueError(f"vertex={v} out of range [1, {n}]")
        return int(deg[int(v)])

    def n_vertices() -> int:
        return n

    def n_edges() -> int:
        return int(m_edges)

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _as_int_color(c) -> Optional[int]:
        try:
            ci = int(c)
        except Exception:
            return None
        if ci < 1:
            return None
        return ci

    def color_conflicts(coloring: dict) -> list:
        """Return list of (u, v) edge pairs whose endpoints share the same
        color in `coloring`. Each edge reported once with u < v. Vertices
        missing from `coloring` are ignored for that edge."""
        out = []
        for (u, v) in edges:
            uu, vv = (u, v) if u < v else (v, u)
            cu = coloring.get(uu)
            cv = coloring.get(vv)
            if cu is None or cv is None:
                continue
            if cu == cv:
                out.append((uu, vv))
        return out

    def is_proper_coloring(coloring: dict) -> bool:
        """True iff every vertex 1..n has a positive integer color assigned
        and no edge has both endpoints the same color."""
        for v in range(1, n + 1):
            c = coloring.get(v)
            if _as_int_color(c) is None:
                return False
        return len(color_conflicts(coloring)) == 0

    def colors_used(coloring: dict) -> set:
        """Set of distinct color values appearing in `coloring`."""
        return {coloring[v] for v in coloring if coloring[v] is not None}

    def saturation(v: int, coloring: dict) -> int:
        """Number of DISTINCT colors among v's already-colored neighbors.
        Used by DSATUR: pick the uncolored vertex with highest saturation."""
        if not (1 <= int(v) <= n):
            raise ValueError(f"vertex={v} out of range [1, {n}]")
        seen = set()
        for u in adj_list[int(v)]:
            c = coloring.get(u)
            if c is not None:
                seen.add(c)
        return len(seen)

    # ==================================================================
    # (3) Construction / improvement
    # ==================================================================
    def _smallest_available_color(v: int, coloring: dict) -> int:
        """Smallest positive integer color not used by any colored neighbor."""
        used = set()
        for u in adj_list[v]:
            c = coloring.get(u)
            if c is not None:
                used.add(c)
        c = 1
        while c in used:
            c += 1
        return c

    def greedy_color(order: Optional[Iterable[int]] = None) -> dict:
        """Greedy (Welsh-Powell when order is by descending degree) coloring:
        process vertices in `order` (default: descending degree, ties by id)
        and assign each the smallest color not used by its already-colored
        neighbors. Returns {vertex -> color}, with every vertex 1..n covered."""
        if order is None:
            order = sorted(range(1, n + 1), key=lambda v: (-deg[v], v))
        coloring: dict[int, int] = {}
        for v in order:
            vi = int(v)
            if not (1 <= vi <= n):
                continue
            coloring[vi] = _smallest_available_color(vi, coloring)
        # Ensure every vertex got one (handles partial / duplicated orders)
        for v in range(1, n + 1):
            if v not in coloring:
                coloring[v] = _smallest_available_color(v, coloring)
        return coloring

    def dsatur_color() -> dict:
        """DSATUR (Brelaz, 1979). Repeatedly pick the uncolored vertex with the
        highest saturation degree (number of distinct colors among colored
        neighbors); break ties by highest plain degree, then smallest id.
        Assign it the smallest color not used by its neighbors. Returns
        {vertex -> color}, every vertex 1..n covered. O(n^2)."""
        coloring: dict[int, int] = {}
        sat: dict[int, set] = {v: set() for v in range(1, n + 1)}
        uncolored = set(range(1, n + 1))
        while uncolored:
            # pick vertex with max saturation, then max degree, then min id
            best = min(
                uncolored,
                key=lambda v: (-len(sat[v]), -deg[v], v),
            )
            c = 1
            neighbor_colors = sat[best]
            while c in neighbor_colors:
                c += 1
            coloring[best] = c
            uncolored.discard(best)
            for u in adj_list[best]:
                if u in uncolored:
                    sat[u].add(c)
        return coloring

    def apply_recolor_vertex(coloring: dict, v: int, new_color: int) -> Optional[dict]:
        """Return a copy of `coloring` with vertex v reassigned to `new_color`,
        IF that keeps the coloring proper (no neighbor of v has new_color).
        Returns None if the change would introduce a conflict, or if inputs
        are invalid. Useful for local-search moves."""
        vi = int(v)
        nc = _as_int_color(new_color)
        if nc is None or not (1 <= vi <= n):
            return None
        for u in adj_list[vi]:
            if coloring.get(u) == nc and u != vi:
                return None
        new = dict(coloring)
        new[vi] = nc
        return new

    def recolor_to_minimize_colors(coloring: dict, max_passes: int = 50) -> dict:
        """Try to reduce the number of distinct colors used. Strategy:
        repeatedly find the color class with the FEWEST vertices and try to
        re-assign each of its vertices to any OTHER color whose vertices are
        all non-adjacent to it. If every vertex in that class can be moved,
        the color is eliminated. Pure local-search; keeps coloring proper.
        Returns a (possibly improved) coloring covering all 1..n vertices."""
        col = dict(coloring)
        # Ensure proper input -- if not, fall back to greedy.
        if not is_proper_coloring(col):
            col = greedy_color()

        for _ in range(int(max_passes)):
            classes: dict[int, list[int]] = {}
            for v, c in col.items():
                classes.setdefault(int(c), []).append(v)
            if len(classes) <= 1:
                break
            # smallest color class first (cheapest to eliminate)
            smallest = min(classes.keys(), key=lambda c: (len(classes[c]), c))
            target_vertices = list(classes[smallest])
            other_colors = [c for c in classes if c != smallest]
            tentative = dict(col)
            ok = True
            for v in target_vertices:
                placed = False
                for nc in other_colors:
                    if all(tentative.get(u) != nc for u in adj_list[v]):
                        tentative[v] = nc
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if ok:
                col = tentative
            else:
                # Couldn't eliminate the smallest class this pass -- stop.
                break
        return col

    # ==================================================================
    # (4) Exact / heavy
    # ==================================================================
    def ilp_chromatic_number(time_limit_s: float = 30.0) -> Optional[dict]:
        """Exact graph-coloring ILP via python-mip / CBC.

        Variables:
          w[c] in {0,1} : color c is used
          x[v,c] in {0,1} : vertex v gets color c
        Objective: minimize sum_c w[c].
        Constraints:
          sum_c x[v,c] = 1        for each vertex v
          x[u,c] + x[v,c] <= 1    for each edge (u,v), each color c
          x[v,c] <= w[c]          for each (v, c)

        Color upper bound = max_degree + 1 (Brooks). Returns the best
        coloring found within `time_limit_s` as {vertex -> color}, or None
        if no feasible solution was returned. Symmetry-breaking: w[c] is
        monotone (w[c] >= w[c+1]) to cut the search tree.
        """
        try:
            from mip import Model, BINARY, MINIMIZE, xsum, OptimizationStatus
        except Exception:
            return None
        if n == 0:
            return {}

        K = max(1, max_deg + 1)  # Brooks upper bound, at least 1 color
        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)

        w = {c: m.add_var(var_type=BINARY, name=f"w[{c}]") for c in range(1, K + 1)}
        x = {(v, c): m.add_var(var_type=BINARY, name=f"x[{v},{c}]")
             for v in range(1, n + 1) for c in range(1, K + 1)}

        m.objective = xsum(w[c] for c in range(1, K + 1))

        # Each vertex gets exactly one color.
        for v in range(1, n + 1):
            m += xsum(x[v, c] for c in range(1, K + 1)) == 1, f"one_color_{v}"

        # Adjacent vertices differ. (Iterate canonical edges only.)
        seen_edges = set()
        for (u, v) in edges:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen_edges or a == b:
                continue
            seen_edges.add((a, b))
            for c in range(1, K + 1):
                m += x[a, c] + x[b, c] <= 1, f"edge_{a}_{b}_c{c}"

        # x uses w.
        for v in range(1, n + 1):
            for c in range(1, K + 1):
                m += x[v, c] <= w[c], f"link_{v}_{c}"

        # Symmetry breaking: prefer lower color indices.
        for c in range(1, K):
            m += w[c] >= w[c + 1], f"sym_{c}"

        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None

        out: dict[int, int] = {}
        for v in range(1, n + 1):
            chosen = None
            for c in range(1, K + 1):
                val = x[v, c].x
                if val is not None and val > 0.5:
                    chosen = c
                    break
            if chosen is None:
                return None  # malformed solution
            out[v] = int(chosen)
        return out

    return {
        # queries
        "adjacency": adjacency,
        "degree": degree,
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        # feasibility
        "color_conflicts": color_conflicts,
        "is_proper_coloring": is_proper_coloring,
        "colors_used": colors_used,
        "saturation": saturation,
        # construction / improvement
        "greedy_color": greedy_color,
        "dsatur_color": dsatur_color,
        "apply_recolor_vertex": apply_recolor_vertex,
        "recolor_to_minimize_colors": recolor_to_minimize_colors,
        # exact
        "ilp_chromatic_number": ilp_chromatic_number,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "adjacency",
        "input": "v: int",
        "output": "list[int]",
        "purpose": (
            "Neighbors of vertex v (1-indexed) as a list, sorted ascending. "
            "Equivalent to list(instance['adjacency'][v]) but cached and "
            "deterministic. Raises if v is out of [1, n]."
        ),
    },
    {
        "name": "degree",
        "input": "v: int",
        "output": "int",
        "purpose": "Degree of vertex v (1-indexed). Precomputed; O(1).",
    },
    {
        "name": "n_vertices",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of vertices in the graph (== instance['n']).",
    },
    {
        "name": "n_edges",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of UNIQUE edges in the graph (each undirected edge counted once).",
    },
    # ----- Feasibility primitives -----
    {
        "name": "color_conflicts",
        "input": "coloring: dict[int, int]",
        "output": "list[tuple[int, int]]",
        "purpose": (
            "List of edges (u, v) with u < v whose endpoints share the same "
            "color in `coloring`. Empty list = proper coloring on the colored "
            "subset. Vertices missing from `coloring` are skipped for that edge."
        ),
    },
    {
        "name": "is_proper_coloring",
        "input": "coloring: dict[int, int]",
        "output": "bool",
        "purpose": (
            "True iff every vertex 1..n has a positive integer color AND no "
            "edge connects two vertices of the same color. Equivalent to a "
            "feasibility check (faster than calling is_feasible)."
        ),
    },
    {
        "name": "colors_used",
        "input": "coloring: dict[int, int]",
        "output": "set[int]",
        "purpose": (
            "Distinct color values appearing in `coloring`. len(colors_used) "
            "is what the evaluator scores when the coloring is proper."
        ),
    },
    {
        "name": "saturation",
        "input": "v: int, coloring: dict[int, int]",
        "output": "int",
        "purpose": (
            "Number of distinct colors among v's ALREADY-colored neighbors. "
            "Core DSATUR primitive: at each step, pick the uncolored vertex "
            "with the highest saturation."
        ),
    },
    # ----- Construction / improvement -----
    {
        "name": "greedy_color",
        "input": "order: Iterable[int] | None = None",
        "output": "dict[int, int]",
        "purpose": (
            "Greedy coloring: process vertices in `order` (default: descending "
            "degree, i.e. Welsh-Powell) and assign each the smallest color not "
            "used by its already-colored neighbors. Returns a proper coloring "
            "for all 1..n. Fast O(n + m) warm start; uses at most max_degree+1 "
            "colors."
        ),
    },
    {
        "name": "dsatur_color",
        "input": "(no args)",
        "output": "dict[int, int]",
        "purpose": (
            "DSATUR algorithm (Brelaz, 1979): repeatedly pick the uncolored "
            "vertex with the highest saturation degree (ties by max degree, "
            "then min id) and assign the smallest available color. Returns a "
            "proper coloring for all 1..n. Typically beats plain greedy by "
            "1-3 colors on dense graphs."
        ),
    },
    {
        "name": "apply_recolor_vertex",
        "input": "coloring: dict[int, int], v: int, new_color: int",
        "output": "dict[int, int] | None",
        "purpose": (
            "If recoloring vertex v to `new_color` keeps the coloring proper "
            "(no neighbor already has `new_color`), return a NEW coloring dict "
            "with the change applied; otherwise return None. Use for local-"
            "search moves; does not mutate the input."
        ),
    },
    {
        "name": "recolor_to_minimize_colors",
        "input": "coloring: dict[int, int], max_passes: int = 50",
        "output": "dict[int, int]",
        "purpose": (
            "Improve a proper coloring by trying to eliminate the SMALLEST "
            "color class: repeatedly attempt to move every vertex of the "
            "least-used color into some other color class without creating "
            "conflicts. If all members can be relocated, that color is gone. "
            "Runs up to `max_passes` rounds. Never increases the color count."
        ),
    },
    # ----- Exact -----
    {
        "name": "ilp_chromatic_number",
        "input": "time_limit_s: float = 30.0",
        "output": "dict[int, int] | None",
        "purpose": (
            "Exact graph-coloring ILP via python-mip/CBC (open-source). "
            "Variables: w[c] (color c used), x[v,c] (v takes color c). "
            "Minimizes sum_c w[c] with the standard constraints plus symmetry-"
            "breaking (w monotone) and a Brooks upper bound of max_degree+1 "
            "colors. Returns the best coloring found within `time_limit_s` as "
            "{vertex -> color}, or None if no feasible solution was returned. "
            "Scales to ~100 vertices; for larger graphs prefer DSATUR + "
            "recolor_to_minimize_colors."
        ),
    },
]
