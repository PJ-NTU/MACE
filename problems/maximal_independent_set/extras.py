"""Per-problem extras for CO-Bench Maximal Independent Set.

Provides primitive building blocks so the LLM can compose construction +
local-search heuristics (min-degree greedy, max-degree removal, k-swap
improvement) and optionally call an exact 0-1 ILP solver. The solution
shape expected by CO-Bench is:
    {'mis_nodes': list[node_id]}
where node_ids are whatever the underlying networkx.Graph uses (typically
0..n-1 ints for ER graphs, but adapters preserve whatever the file has).

Tool groups:
  (1) Queries:        adjacency, degree, n_vertices, n_edges
  (2) Feasibility:    is_independent_set, size_of_set, forbidden_by
  (3) Construction:   greedy_min_degree, greedy_max_degree_removal
  (4) Improvement:    apply_swap_2_for_1, apply_local_swap
  (5) Exact (heavy):  ilp_max_independent_set

All are optional. The LLM may use any subset or write everything from scratch.
"""
from __future__ import annotations
import random
import time
from typing import Iterable, Optional


def extra_tools(instance: dict) -> dict:
    """Factory: returns MIS-specific tool callables for `instance`.

    Instance schema (from CO-Bench Maximal independent set load_data):
      - name:  str, instance identifier
      - graph: networkx.Graph (undirected, simple)
    """
    G = instance["graph"]

    # Snapshot the node list and adjacency once. Using a set-based adjacency
    # gives O(1) edge / neighbor membership for the local-search / swap
    # primitives below. networkx graphs are mutable; the snapshot insulates
    # the helpers from any incidental mutation by the LLM.
    node_list = list(G.nodes())
    node_set = set(node_list)
    n = len(node_list)
    adj: dict = {v: set(G.neighbors(v)) for v in node_list}
    # Self-loops would break independence-checks; strip them defensively.
    for v in node_list:
        adj[v].discard(v)
    deg = {v: len(adj[v]) for v in node_list}
    m_edges = sum(deg.values()) // 2

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def adjacency(v) -> list:
        """Neighbors of v as a list. Raises if v not in the graph."""
        if v not in node_set:
            raise ValueError(f"vertex {v!r} not in graph")
        return list(adj[v])

    def degree(v) -> int:
        if v not in node_set:
            raise ValueError(f"vertex {v!r} not in graph")
        return int(deg[v])

    def n_vertices() -> int:
        return int(n)

    def n_edges() -> int:
        return int(m_edges)

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def is_independent_set(selected: Iterable) -> bool:
        """True iff `selected` contains no duplicates, every element is in
        the graph, and no two elements are adjacent (no edge between them)."""
        sel = list(selected)
        if len(sel) != len(set(sel)):
            return False
        for v in sel:
            if v not in node_set:
                return False
        sset = set(sel)
        for v in sel:
            # Any neighbor of v also in the set => not independent.
            if adj[v] & sset:
                return False
        return True

    def size_of_set(selected: Iterable) -> int:
        """Number of UNIQUE in-graph vertices in `selected`."""
        return len({v for v in selected if v in node_set})

    def forbidden_by(selected: Iterable) -> set:
        """Vertices that cannot be added to `selected` without breaking
        independence: the union of `selected` itself plus all neighbors of
        any vertex in `selected`."""
        out: set = set()
        for v in selected:
            if v in node_set:
                out.add(v)
                out |= adj[v]
        return out

    # ==================================================================
    # (3) Construction heuristics
    # ==================================================================
    def greedy_min_degree() -> list:
        """Classic min-degree greedy for MIS: at each step, pick the
        remaining vertex of MINIMUM degree (in the residual graph), add it
        to the independent set, then delete it AND its neighbors. Repeat
        until no vertices remain. Returns the independent set as a list.
        O((n + m) log n). Gives a Delta+1 approximation and is a strong
        warm start for local search."""
        # Mutable residual degree map.
        residual_deg = dict(deg)
        alive = set(node_list)
        chosen: list = []
        while alive:
            # Pick the alive vertex of smallest residual degree.
            v = min(alive, key=lambda u: (residual_deg[u], _stable_key(u)))
            chosen.append(v)
            # Remove v and all its alive neighbors.
            to_remove = {v}
            for u in adj[v]:
                if u in alive:
                    to_remove.add(u)
            # Update residual degrees for vertices adjacent to anything removed.
            affected: set = set()
            for r in to_remove:
                for w in adj[r]:
                    if w in alive and w not in to_remove:
                        affected.add(w)
            alive -= to_remove
            for w in affected:
                # Count alive neighbors of w.
                residual_deg[w] = sum(1 for x in adj[w] if x in alive)
        return chosen

    def greedy_max_degree_removal() -> list:
        """Complementary heuristic: iteratively REMOVE the vertex of
        maximum residual degree until the remaining graph has no edges.
        The remaining vertices form an independent set. Tends to do well
        on dense or non-uniform graphs where min-degree greedy locks in
        suboptimal early choices. Returns the independent set as a list."""
        residual_deg = dict(deg)
        alive = set(node_list)
        # Count alive edges as we go.
        edge_count = m_edges
        while edge_count > 0 and alive:
            v = max(alive, key=lambda u: (residual_deg[u], _stable_key(u)))
            # Remove v; decrement degrees of its alive neighbors.
            for w in adj[v]:
                if w in alive:
                    residual_deg[w] -= 1
                    edge_count -= 1
            alive.discard(v)
            residual_deg.pop(v, None)
        return list(alive)

    # ==================================================================
    # (4) Improvement primitives
    # ==================================================================
    def apply_swap_2_for_1(selected: Iterable) -> list:
        """(2,1)-swap improvement: try to replace ONE vertex v in `selected`
        with TWO non-adjacent neighbors of v that are otherwise free
        (i.e., adjacent to NO other vertex in the current set). Each such
        swap increases |S| by 1. Repeats until no further (2,1)-swap is
        possible. Returns the improved independent set as a list.

        Precondition: `selected` should already be a (proper) independent
        set; if it isn't, the function still runs but never violates
        independence in its output."""
        cur = [v for v in selected if v in node_set]
        cur_set = set(cur)
        # Ensure starting feasibility -- silently drop conflicting picks.
        cleaned: list = []
        cleaned_set: set = set()
        for v in cur:
            if v in cleaned_set:
                continue
            if cleaned_set & adj[v]:
                continue
            cleaned.append(v)
            cleaned_set.add(v)
        cur, cur_set = cleaned, cleaned_set

        improved = True
        while improved:
            improved = False
            for v in list(cur):
                # Candidate replacements: neighbors of v whose ONLY
                # in-set neighbor is v (so removing v frees them).
                free_neighbors = []
                for u in adj[v]:
                    if u in cur_set:
                        continue
                    # u must have no in-set neighbor other than v.
                    others = adj[u] & cur_set
                    if others == {v}:
                        free_neighbors.append(u)
                # Look for two non-adjacent free neighbors.
                found = None
                L = len(free_neighbors)
                for i in range(L):
                    a = free_neighbors[i]
                    a_adj = adj[a]
                    for j in range(i + 1, L):
                        b = free_neighbors[j]
                        if b not in a_adj:
                            found = (a, b)
                            break
                    if found is not None:
                        break
                if found is not None:
                    a, b = found
                    cur_set.discard(v)
                    cur_set.add(a)
                    cur_set.add(b)
                    # Rebuild order: stable removal of v, append a, b.
                    cur = [x for x in cur if x != v] + [a, b]
                    improved = True
                    break  # restart outer loop
        return cur

    def apply_local_swap(selected: Iterable, t_limit: float = 5.0) -> list:
        """Combined local search: alternates (2,1)-swap improvement with
        randomized perturbation. Each perturbation drops a random in-set
        vertex (opening neighborhood) and re-runs the (2,1)-swap, keeping
        the best independent set seen. Time-bounded by `t_limit` (seconds).
        Returns the best independent set found."""
        cur = apply_swap_2_for_1(selected)
        best = list(cur)
        if not cur:
            return best
        t0 = time.time()
        safety = 0.05
        rng = random.Random(0xC0BE17)
        while (time.time() - t0) < float(t_limit) - safety:
            if not cur:
                break
            # Drop one random vertex, then re-improve.
            victim = rng.choice(cur)
            tentative = [x for x in cur if x != victim]
            # Try to expand: any vertex with no in-set neighbor is addable.
            cur_set = set(tentative)
            forbid = set(tentative)
            for u in tentative:
                forbid |= adj[u]
            # Add free vertices in a shuffled order for diversification.
            free = [v for v in node_list if v not in forbid]
            rng.shuffle(free)
            for v in free:
                if not (adj[v] & cur_set):
                    cur_set.add(v)
                    tentative.append(v)
            cur = apply_swap_2_for_1(tentative)
            if len(cur) > len(best):
                best = list(cur)
        return best

    # ==================================================================
    # (5) Exact / heavy
    # ==================================================================
    def ilp_max_independent_set(time_limit_s: float = 30.0) -> Optional[list]:
        """Exact 0-1 ILP for Maximum Independent Set via python-mip / CBC.

        Variables: x[v] in {0,1} for each vertex v.
        Objective: MAXIMIZE sum_v x[v].
        Constraints: x[u] + x[v] <= 1 for each edge (u, v).

        Returns the best independent set found within `time_limit_s` as a
        list of node ids, or None if the solver was unavailable / returned
        no feasible solution. Scales practically to a few hundred vertices
        on sparse graphs; for the dense / large instances in CO-Bench,
        prefer greedy + apply_local_swap."""
        try:
            from mip import Model, BINARY, MAXIMIZE, xsum, OptimizationStatus
        except Exception:
            return None
        if n == 0:
            return []
        m = Model(sense=MAXIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)
        x = {v: m.add_var(var_type=BINARY, name=f"x[{v}]") for v in node_list}
        m.objective = xsum(x[v] for v in node_list)
        # Edges: iterate each unique edge once.
        added = set()
        for u in node_list:
            for w in adj[u]:
                key = (u, w) if _stable_key(u) <= _stable_key(w) else (w, u)
                if key in added or key[0] == key[1]:
                    continue
                added.add(key)
                m += x[key[0]] + x[key[1]] <= 1, f"edge_{len(added)}"
        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None
        out = []
        for v in node_list:
            val = x[v].x
            if val is not None and val > 0.5:
                out.append(v)
        return out

    return {
        # queries
        "adjacency": adjacency,
        "degree": degree,
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        # feasibility
        "is_independent_set": is_independent_set,
        "size_of_set": size_of_set,
        "forbidden_by": forbidden_by,
        # construction
        "greedy_min_degree": greedy_min_degree,
        "greedy_max_degree_removal": greedy_max_degree_removal,
        # improvement
        "apply_swap_2_for_1": apply_swap_2_for_1,
        "apply_local_swap": apply_local_swap,
        # exact
        "ilp_max_independent_set": ilp_max_independent_set,
    }


def _stable_key(v):
    """Sortable tie-breaker that works for ints, strings, tuples, etc.
    networkx node ids are commonly ints but can be arbitrary hashables;
    we fall back to repr() so min()/max() never raises TypeError."""
    try:
        # Cheap fast-path: numeric ids sort numerically.
        return (0, v)
    except Exception:
        return (1, repr(v))


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "adjacency",
        "input": "v",
        "output": "list",
        "purpose": (
            "Neighbors of vertex v as a list. v must be a node id present in "
            "instance['graph']. O(deg(v))."
        ),
    },
    {
        "name": "degree",
        "input": "v",
        "output": "int",
        "purpose": "Degree of vertex v. Precomputed; O(1).",
    },
    {
        "name": "n_vertices",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of vertices in the graph.",
    },
    {
        "name": "n_edges",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of unique undirected edges in the graph.",
    },
    # ----- Feasibility primitives -----
    {
        "name": "is_independent_set",
        "input": "selected: Iterable",
        "output": "bool",
        "purpose": (
            "True iff `selected` is duplicate-free, all elements are nodes in "
            "the graph, AND no two elements are connected by an edge. "
            "Equivalent feasibility check to is_feasible but faster (no "
            "exception path)."
        ),
    },
    {
        "name": "size_of_set",
        "input": "selected: Iterable",
        "output": "int",
        "purpose": (
            "Count of UNIQUE in-graph nodes in `selected`. This is the "
            "value the evaluator scores when the selection is a valid "
            "independent set (higher = better)."
        ),
    },
    {
        "name": "forbidden_by",
        "input": "selected: Iterable",
        "output": "set",
        "purpose": (
            "Vertices that cannot be added to `selected` without breaking "
            "independence: the union of `selected` itself plus all neighbors "
            "of any vertex already in `selected`. Useful for incremental "
            "construction: addable = all_vertices - forbidden_by(selected)."
        ),
    },
    # ----- Construction -----
    {
        "name": "greedy_min_degree",
        "input": "(no args)",
        "output": "list",
        "purpose": (
            "Min-degree greedy: at each step pick the residual vertex of "
            "MINIMUM degree, add it to the set, delete it and its neighbors, "
            "repeat. Returns an independent set (Delta+1 approximation). "
            "Strong warm start for local search; O((n+m) log n)."
        ),
    },
    {
        "name": "greedy_max_degree_removal",
        "input": "(no args)",
        "output": "list",
        "purpose": (
            "Complementary heuristic: iteratively REMOVE the highest-degree "
            "vertex from the residual graph until no edges remain; the "
            "survivors form an independent set. Often beats min-degree "
            "greedy on dense / power-law graphs."
        ),
    },
    # ----- Improvement -----
    {
        "name": "apply_swap_2_for_1",
        "input": "selected: Iterable",
        "output": "list",
        "purpose": (
            "(2,1)-swap local search: repeatedly try to replace ONE vertex v "
            "in the set with TWO non-adjacent neighbors of v whose only "
            "in-set neighbor is v itself. Each accepted swap grows the set "
            "by 1. Runs until no such swap exists. Pure local move; never "
            "produces an infeasible set. O(n * deg_max^2) per pass."
        ),
    },
    {
        "name": "apply_local_swap",
        "input": "selected: Iterable, t_limit: float = 5.0",
        "output": "list",
        "purpose": (
            "Time-bounded local search: alternates (2,1)-swap improvement "
            "with randomized perturbation (drop one random in-set vertex, "
            "greedily re-fill any newly-free positions, re-improve). Keeps "
            "the best independent set found within `t_limit` seconds. "
            "Returns the best set as a list."
        ),
    },
    # ----- Exact -----
    {
        "name": "ilp_max_independent_set",
        "input": "time_limit_s: float = 30.0",
        "output": "list | None",
        "purpose": (
            "Exact 0-1 ILP via python-mip / CBC. Variables x[v] in {0,1}; "
            "maximize sum_v x[v]; for each edge (u,v): x[u] + x[v] <= 1. "
            "Returns the best independent set found within `time_limit_s` "
            "as a list of node ids, or None if the solver was unavailable "
            "or returned no feasible solution. Scales to a few hundred "
            "vertices on sparse graphs; on the CO-Bench ER_700_800 "
            "instances expect to hit the time limit, so still useful as a "
            "warm-restart with `time_limit_s` set generously."
        ),
    },
]
