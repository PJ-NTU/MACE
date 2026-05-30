"""Per-problem extras for CO-Bench Resource Constrained Shortest Path.

Provides building-block tools so the LLM can compose construction +
exact / heuristic algorithms for RCSP. Vertices are 1-indexed; source=1,
target=n; each arc carries a length (cost) plus K resource consumptions,
each vertex (including endpoints) also carries K vertex resources; a path
is feasible iff for every resource k the cumulative consumption (sum of
vertex_resources over visited vertices plus arc_resources over traversed
arcs) lies in [lower_bounds[k], upper_bounds[k]].

Tool groups:
  (1) Queries:        edge_length, edge_resource, resource_budget,
                      n_vertices, n_edges, source, target
  (2) Feasibility:    path_length, path_resources, is_feasible_path
  (3) Construction:   dijkstra_pure_length, greedy_extend_path,
                      label_setting_pareto
  (4) Exact (heavy):  ilp_rcsp

All are optional; the LLM may use any subset.
"""
from __future__ import annotations
import heapq
import time
from typing import Optional


def extra_tools(instance: dict) -> dict:
    """Factory: returns RCSP-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench RCSP load_data):
      - n:                int, number of vertices (1..n).
      - m:                int, number of arcs.
      - K:                int, number of resources.
      - lower_bounds:     list[float] of length K.
      - upper_bounds:     list[float] of length K.
      - vertex_resources: list (length n) of K-length lists.
      - graph:            dict[int -> list[(end_vertex, cost, [res_k])]]
                          1-indexed adjacency.
    """
    n: int = int(instance["n"])
    m: int = int(instance["m"])
    K: int = int(instance["K"])
    lower_bounds = [float(x) for x in instance["lower_bounds"]]
    upper_bounds = [float(x) for x in instance["upper_bounds"]]
    vertex_resources = [
        [float(x) for x in row] for row in instance["vertex_resources"]
    ]
    raw_graph = instance["graph"]

    SRC, TGT = 1, n

    # Normalize the graph once: keep first occurrence of each (u, v) arc but
    # remember the full list so that parallel arcs (different cost/resources
    # between the same pair) are not silently lost. For edge_length /
    # edge_resource we expose the BEST (smallest cost) arc among parallels.
    # Algorithms that traverse arcs internally use the full list.
    adj: dict[int, list[tuple[int, float, list[float]]]] = {
        v: [] for v in range(1, n + 1)
    }
    for u in range(1, n + 1):
        for tup in raw_graph.get(u, []):
            v = int(tup[0])
            c = float(tup[1])
            r = [float(x) for x in tup[2]]
            adj[u].append((v, c, r))

    # Best (cheapest) arc for each (u, v) pair -- used by the simple queries.
    best_arc: dict[tuple[int, int], tuple[float, list[float]]] = {}
    for u in range(1, n + 1):
        for (v, c, r) in adj[u]:
            key = (u, v)
            if key not in best_arc or c < best_arc[key][0]:
                best_arc[key] = (c, r)

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def edge_length(u: int, v: int) -> Optional[float]:
        """Length (cost) of the cheapest arc u->v, or None if no arc exists."""
        e = best_arc.get((int(u), int(v)))
        return None if e is None else float(e[0])

    def edge_resource(u: int, v: int, r: int) -> Optional[float]:
        """Consumption of resource `r` on the cheapest arc u->v (matches
        edge_length's chosen arc). None if no arc exists. r is 0-indexed."""
        e = best_arc.get((int(u), int(v)))
        if e is None:
            return None
        ri = int(r)
        if not (0 <= ri < K):
            raise ValueError(f"resource index r={r} out of range [0, {K})")
        return float(e[1][ri])

    def resource_budget(r: int) -> tuple[float, float]:
        """Return (lower_bound, upper_bound) for resource r (0-indexed)."""
        ri = int(r)
        if not (0 <= ri < K):
            raise ValueError(f"resource index r={r} out of range [0, {K})")
        return (lower_bounds[ri], upper_bounds[ri])

    def n_vertices() -> int:
        return n

    def n_edges() -> int:
        return m

    def source() -> int:
        return SRC

    def target() -> int:
        return TGT

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _validate_path(path) -> Optional[str]:
        if not path:
            return "empty path"
        for v in path:
            if not (1 <= int(v) <= n):
                return f"vertex {v} out of range [1, {n}]"
        for i in range(len(path) - 1):
            u, v = int(path[i]), int(path[i + 1])
            if (u, v) not in best_arc:
                return f"no arc from {u} to {v}"
        return None

    def path_length(path) -> float:
        """Sum of arc costs along `path` (cheapest parallel arc per consecutive
        pair). Raises ValueError if any consecutive pair has no arc."""
        err = _validate_path(path)
        if err is not None:
            raise ValueError(err)
        total = 0.0
        for i in range(len(path) - 1):
            u, v = int(path[i]), int(path[i + 1])
            total += best_arc[(u, v)][0]
        return float(total)

    def path_resources(path) -> list[float]:
        """Per-resource total consumption along `path`: vertex_resources for
        every vertex in path PLUS arc resources for the cheapest arc on each
        consecutive pair. Returns a length-K list. Mirrors eval_func."""
        err = _validate_path(path)
        if err is not None:
            raise ValueError(err)
        totals = [0.0] * K
        for v in path:
            for k in range(K):
                totals[k] += vertex_resources[int(v) - 1][k]
        for i in range(len(path) - 1):
            u, v = int(path[i]), int(path[i + 1])
            r = best_arc[(u, v)][1]
            for k in range(K):
                totals[k] += r[k]
        return totals

    def is_feasible_path(path) -> tuple[bool, Optional[str]]:
        """(True, None) if `path` starts at 1, ends at n, every consecutive
        pair has an arc, and total resource consumption lies in
        [lower_bounds[k], upper_bounds[k]] for every k. Otherwise
        (False, error_message). Cheaper than tools['is_feasible'] when you
        already have the path."""
        if not path or int(path[0]) != SRC or int(path[-1]) != TGT:
            return False, f"path must start at {SRC} and end at {TGT}"
        err = _validate_path(path)
        if err is not None:
            return False, err
        totals = path_resources(path)
        for k in range(K):
            if totals[k] < lower_bounds[k] - 1e-6:
                return False, (f"resource {k} = {totals[k]} below lower "
                               f"bound {lower_bounds[k]}")
            if totals[k] > upper_bounds[k] + 1e-6:
                return False, (f"resource {k} = {totals[k]} above upper "
                               f"bound {upper_bounds[k]}")
        return True, None

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def dijkstra_pure_length() -> Optional[list]:
        """Plain Dijkstra from SRC=1 to TGT=n IGNORING resource constraints.
        Returns the minimum-length path as a list of vertices, or None if TGT
        is unreachable. The returned path's length is a LOWER BOUND on any
        feasible RCSP solution; the path itself may violate resource bounds."""
        INF = float("inf")
        dist = {v: INF for v in range(1, n + 1)}
        prev = {v: None for v in range(1, n + 1)}
        dist[SRC] = 0.0
        pq: list[tuple[float, int]] = [(0.0, SRC)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-12:
                continue
            if u == TGT:
                break
            for (v, c, _r) in adj[u]:
                nd = d + c
                if nd + 1e-12 < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[TGT] == INF:
            return None
        path = [TGT]
        while path[-1] != SRC:
            p = prev[path[-1]]
            if p is None:
                return None
            path.append(p)
        path.reverse()
        return path

    def greedy_extend_path(time_limit_s: float = 2.0) -> Optional[list]:
        """Greedy forward construction with resource-aware pruning. At each
        step, from the current vertex, pick the outgoing arc that minimizes
        cost subject to: the running resource totals after taking that arc
        must leave enough headroom under upper_bounds[k] (we estimate
        per-vertex resource >= 0 and assume at least one more vertex). Stops
        if it reaches TGT; otherwise returns None. Cheap, often produces a
        feasible starting solution but no optimality guarantee."""
        t0 = time.time()
        cur = SRC
        path = [SRC]
        # running totals already include SRC's vertex_resources
        totals = [vertex_resources[SRC - 1][k] for k in range(K)]
        visited = {SRC}
        while cur != TGT:
            if time.time() - t0 > time_limit_s:
                return None
            best: Optional[tuple[float, int, list[float]]] = None
            for (v, c, r) in adj[cur]:
                if v in visited:
                    continue
                # tentative totals after going cur -> v (adds arc res + vertex_resources[v])
                ok = True
                tent = list(totals)
                for k in range(K):
                    tent[k] += r[k] + vertex_resources[v - 1][k]
                    if tent[k] > upper_bounds[k] + 1e-6:
                        ok = False
                        break
                if not ok:
                    continue
                if best is None or c < best[0]:
                    best = (c, v, tent)
            if best is None:
                return None
            _, v, tent = best
            path.append(v)
            visited.add(v)
            totals = tent
            cur = v
        # Verify lower bounds at the end (a too-frugal path may underconsume)
        for k in range(K):
            if totals[k] < lower_bounds[k] - 1e-6:
                return None
        return path

    def label_setting_pareto(
        time_limit_s: float = 5.0,
        k_labels: int = 200,
    ) -> Optional[list]:
        """Pareto label-setting (classic RCSP exact algorithm, with a label-
        count cap for tractability).

        A label at vertex u is (cost, [resource_totals], prev_label_idx, u).
        We expand labels in order of cost (Dijkstra-style). When extending
        u->v via arc (cost c, res r), the new label adds c to cost, r[k] +
        vertex_resources[v-1][k] to each resource. We prune labels that
        exceed any upper bound. At each vertex we keep only PARETO-OPTIMAL
        labels (no other label has <= cost AND <= every resource, with at
        least one strictly <). If more than `k_labels` labels accumulate at
        a vertex, we keep the k_labels cheapest by cost (this makes the
        algorithm heuristic at the cap; raise k_labels for accuracy).

        Returns the cheapest path at TGT that ALSO satisfies the LOWER
        bounds, or None if no feasible label was found within time_limit_s.
        With k_labels large enough and time enough, this is EXACT.
        """
        t0 = time.time()
        # labels[v] = list of (cost, resources_tuple, parent_idx, prev_v)
        # `parent_idx` indexes labels[prev_v] of the predecessor; chain to root.
        labels: dict[int, list[tuple[float, tuple, int, int]]] = {
            v: [] for v in range(1, n + 1)
        }
        # Initial label at SRC: cost 0, resources = vertex_resources[SRC-1]
        init_res = tuple(float(vertex_resources[SRC - 1][k]) for k in range(K))
        # Reject immediately if even the initial vertex exceeds upper bounds.
        for k in range(K):
            if init_res[k] > upper_bounds[k] + 1e-6:
                return None
        labels[SRC].append((0.0, init_res, -1, -1))

        # Priority queue: (cost, vertex, label_idx_in_labels[vertex])
        pq: list[tuple[float, int, int]] = [(0.0, SRC, 0)]

        def _dominates(a: tuple, b: tuple) -> bool:
            """a dominates b iff a.cost <= b.cost AND a.res[k] <= b.res[k]
            for all k AND at least one strict. (a, b are (cost, res, ...))."""
            ac, ar = a[0], a[1]
            bc, br = b[0], b[1]
            if ac > bc + 1e-12:
                return False
            for k in range(K):
                if ar[k] > br[k] + 1e-12:
                    return False
            # at least one strict
            if ac + 1e-12 < bc:
                return True
            for k in range(K):
                if ar[k] + 1e-12 < br[k]:
                    return True
            return False

        best_tgt: Optional[tuple[float, tuple, int, int]] = None

        while pq:
            if time.time() - t0 > time_limit_s:
                break
            c, u, idx = heapq.heappop(pq)
            # The label might have been pruned post-insertion; check it's still
            # in labels[u] at idx and unchanged in cost.
            if idx >= len(labels[u]):
                continue
            lab = labels[u][idx]
            if abs(lab[0] - c) > 1e-9:
                continue
            if u == TGT:
                # check lower bounds
                ok = True
                for k in range(K):
                    if lab[1][k] < lower_bounds[k] - 1e-6:
                        ok = False
                        break
                if ok and (best_tgt is None or lab[0] < best_tgt[0]):
                    best_tgt = lab
                # don't expand past TGT
                continue
            # prune if cost already worse than the best feasible target found
            if best_tgt is not None and lab[0] >= best_tgt[0] - 1e-12:
                continue
            for (v, arc_c, arc_r) in adj[u]:
                new_cost = lab[0] + arc_c
                new_res = list(lab[1])
                ok = True
                for k in range(K):
                    new_res[k] += arc_r[k] + vertex_resources[v - 1][k]
                    if new_res[k] > upper_bounds[k] + 1e-6:
                        ok = False
                        break
                if not ok:
                    continue
                if best_tgt is not None and new_cost >= best_tgt[0] - 1e-12:
                    continue
                new_res_t = tuple(new_res)
                new_lab = (new_cost, new_res_t, idx, u)

                # Dominance check at v
                dominated = False
                kept: list[tuple] = []
                for old in labels[v]:
                    if _dominates(old, new_lab):
                        dominated = True
                        break
                    if not _dominates(new_lab, old):
                        kept.append(old)
                if dominated:
                    continue
                kept.append(new_lab)
                # Cap at k_labels: keep the k_labels cheapest by cost.
                if len(kept) > int(k_labels):
                    kept.sort(key=lambda L: L[0])
                    kept = kept[: int(k_labels)]
                labels[v] = kept
                # Find the new index of new_lab in labels[v] for queue ref.
                # (Equality on the tuple identifies it uniquely.)
                try:
                    new_idx = labels[v].index(new_lab)
                except ValueError:
                    # got bumped out by the cap; skip
                    continue
                heapq.heappush(pq, (new_cost, v, new_idx))

        if best_tgt is None:
            return None
        # Reconstruct path by walking the parent chain. Each label stores
        # (cost, res, parent_idx_in_labels[prev_v], prev_v). We rebuild by
        # looking up labels[prev_v][parent_idx] iteratively until prev_v == -1.
        # But once we replace labels[v] (pruning), parent indices may become
        # stale; in practice, parents we point to from a label that survived
        # were themselves accepted earlier and may have been moved. As a
        # robust fallback, rebuild by re-running a backward search: among the
        # predecessors u of TGT, find one whose label matches lab.cost - arc_c
        # and lab.res - (arc_r + vertex_resources[TGT-1]). Iterate.
        path = [TGT]
        cur_cost = best_tgt[0]
        cur_res = list(best_tgt[1])
        cur = TGT
        # Build reverse adjacency for the backward walk.
        rev: dict[int, list[tuple[int, float, list[float]]]] = {
            v: [] for v in range(1, n + 1)
        }
        for u in range(1, n + 1):
            for (v, c, r) in adj[u]:
                rev[v].append((u, c, r))
        guard = 0
        while cur != SRC and guard < n + 5:
            guard += 1
            found = False
            for (u, arc_c, arc_r) in rev[cur]:
                # If we used this arc to enter cur, then the predecessor
                # label at u should have cost = cur_cost - arc_c and
                # resources = cur_res - arc_r - vertex_resources[cur-1].
                pred_cost = cur_cost - arc_c
                if pred_cost < -1e-6:
                    continue
                pred_res = [cur_res[k] - arc_r[k] - vertex_resources[cur - 1][k]
                            for k in range(K)]
                if any(x < -1e-6 for x in pred_res):
                    continue
                # Search labels[u] for a match.
                matched = False
                for old in labels[u]:
                    if abs(old[0] - pred_cost) > 1e-6:
                        continue
                    ok = True
                    for k in range(K):
                        if abs(old[1][k] - pred_res[k]) > 1e-6:
                            ok = False
                            break
                    if ok:
                        matched = True
                        break
                if matched:
                    path.append(u)
                    cur_cost = pred_cost
                    cur_res = pred_res
                    cur = u
                    found = True
                    break
            if not found:
                return None
        if cur != SRC:
            return None
        path.reverse()
        return path

    # ==================================================================
    # (4) Exact / heavy: ILP
    # ==================================================================
    def ilp_rcsp(time_limit_s: float = 30.0) -> Optional[list]:
        """Exact RCSP via python-mip / CBC.

        Variables: x[u, v, a] in {0, 1} for each parallel arc a between u, v
        (using arc index in adj[u]). Standard SP flow conservation:
          sum_out(SRC) - sum_in(SRC) = 1
          sum_out(TGT) - sum_in(TGT) = -1
          sum_out(v)   - sum_in(v)   = 0   for every other v
        Resource constraints:
          For each k: lower_bounds[k] <=
              sum over chosen arcs of arc_res[k]
              + sum over visited vertices of vertex_resources[v-1][k]
              <= upper_bounds[k]
        where 'visited vertex v' means SRC, TGT, or v has incoming flow >= 1.

        Returns the optimal path as a list of vertices, or None if infeasible
        / no solution within `time_limit_s`. Best for small-to-medium graphs
        (a few hundred vertices, a few thousand arcs).
        """
        try:
            from mip import Model, BINARY, MINIMIZE, xsum, OptimizationStatus
        except Exception:
            return None
        if n < 2:
            return None
        mdl = Model(sense=MINIMIZE)
        mdl.verbose = 0
        mdl.max_seconds = float(time_limit_s)

        # x[(u, a)] = 1 if the a-th outgoing arc of u is used. We index by
        # (u, arc_position_in_adj[u]).
        x: dict[tuple[int, int], object] = {}
        # y[v] = 1 if vertex v is visited (used to charge vertex_resources).
        y: dict[int, object] = {}
        for v in range(1, n + 1):
            y[v] = mdl.add_var(var_type=BINARY, name=f"y[{v}]")
            for a_idx, _ in enumerate(adj[v]):
                x[(v, a_idx)] = mdl.add_var(var_type=BINARY, name=f"x[{v},{a_idx}]")

        # Objective: minimize total arc cost.
        mdl.objective = xsum(
            adj[u][a_idx][1] * x[(u, a_idx)]
            for (u, a_idx) in x
        )

        # Flow conservation.
        # For each vertex v: flow_out(v) - flow_in(v) = (1 if v==SRC, -1 if v==TGT, 0 else)
        for v in range(1, n + 1):
            flow_out = xsum(x[(v, a_idx)] for a_idx in range(len(adj[v])))
            in_arcs = []
            for u in range(1, n + 1):
                for a_idx, (vv, _c, _r) in enumerate(adj[u]):
                    if vv == v:
                        in_arcs.append(x[(u, a_idx)])
            flow_in = xsum(in_arcs) if in_arcs else 0
            if v == SRC:
                mdl += flow_out - flow_in == 1, f"flow_{v}"
            elif v == TGT:
                mdl += flow_out - flow_in == -1, f"flow_{v}"
            else:
                mdl += flow_out - flow_in == 0, f"flow_{v}"

        # Link y[v] to "v is visited".
        # SRC and TGT are always visited.
        mdl += y[SRC] == 1, "src_visited"
        mdl += y[TGT] == 1, "tgt_visited"
        # For other vertices, y[v] = flow_in(v) (which is 0 or 1 due to SP
        # structure on a simple path; flow conservation ensures it's <=1).
        for v in range(1, n + 1):
            if v == SRC or v == TGT:
                continue
            in_arcs = []
            for u in range(1, n + 1):
                for a_idx, (vv, _c, _r) in enumerate(adj[u]):
                    if vv == v:
                        in_arcs.append(x[(u, a_idx)])
            if in_arcs:
                mdl += y[v] == xsum(in_arcs), f"y_link_{v}"
            else:
                mdl += y[v] == 0, f"y_zero_{v}"

        # Resource constraints.
        for k in range(K):
            arc_term = xsum(
                adj[u][a_idx][2][k] * x[(u, a_idx)]
                for (u, a_idx) in x
            )
            vert_term = xsum(
                float(vertex_resources[v - 1][k]) * y[v]
                for v in range(1, n + 1)
            )
            mdl += arc_term + vert_term <= upper_bounds[k] + 1e-6, f"res_ub_{k}"
            mdl += arc_term + vert_term >= lower_bounds[k] - 1e-6, f"res_lb_{k}"

        status = mdl.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if mdl.num_solutions < 1:
            return None

        # Reconstruct the path by walking selected arcs starting at SRC.
        used: dict[int, int] = {}
        for (u, a_idx), var in x.items():
            val = var.x
            if val is not None and val > 0.5:
                # Note: on a simple s-t path each vertex u != TGT has at most
                # one outgoing chosen arc, so this assignment is unambiguous.
                used[u] = a_idx
        path = [SRC]
        cur = SRC
        guard = 0
        while cur != TGT and guard < n + 5:
            guard += 1
            if cur not in used:
                return None
            (v, _c, _r) = adj[cur][used[cur]]
            path.append(v)
            cur = v
        if cur != TGT:
            return None
        return path

    return {
        # queries
        "edge_length": edge_length,
        "edge_resource": edge_resource,
        "resource_budget": resource_budget,
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        "source": source,
        "target": target,
        # feasibility
        "path_length": path_length,
        "path_resources": path_resources,
        "is_feasible_path": is_feasible_path,
        # construction
        "dijkstra_pure_length": dijkstra_pure_length,
        "greedy_extend_path": greedy_extend_path,
        "label_setting_pareto": label_setting_pareto,
        # exact
        "ilp_rcsp": ilp_rcsp,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "edge_length",
        "input": "u: int, v: int",
        "output": "float | None",
        "purpose": (
            "Cost (length) of the cheapest arc u -> v, or None if no such arc "
            "exists. If parallel arcs exist between u and v, this returns the "
            "minimum-cost one (the same one used by path_length / path_resources)."
        ),
    },
    {
        "name": "edge_resource",
        "input": "u: int, v: int, r: int",
        "output": "float | None",
        "purpose": (
            "Consumption of resource r (0-indexed) on the cheapest arc u -> v "
            "(matches edge_length's chosen arc). None if no arc exists; raises "
            "if r is out of [0, K)."
        ),
    },
    {
        "name": "resource_budget",
        "input": "r: int",
        "output": "tuple[float, float]",
        "purpose": (
            "(lower_bound, upper_bound) for resource r (0-indexed). The total "
            "consumption of resource r along a feasible path must lie in this "
            "interval."
        ),
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
        "purpose": "Number of arcs in the graph (== instance['m']).",
    },
    {
        "name": "source",
        "input": "(no args)",
        "output": "int",
        "purpose": "Source vertex (always 1 in CO-Bench RCSP).",
    },
    {
        "name": "target",
        "input": "(no args)",
        "output": "int",
        "purpose": "Target vertex (always n in CO-Bench RCSP).",
    },
    # ----- Feasibility primitives -----
    {
        "name": "path_length",
        "input": "path: list[int]",
        "output": "float",
        "purpose": (
            "Sum of arc costs along `path` (cheapest parallel arc per "
            "consecutive pair). Raises ValueError if any consecutive pair has "
            "no arc or a vertex is out of range."
        ),
    },
    {
        "name": "path_resources",
        "input": "path: list[int]",
        "output": "list[float]",
        "purpose": (
            "Per-resource total consumption along `path` (length K). Sums "
            "vertex_resources for every vertex in path PLUS arc resources for "
            "the cheapest arc on each consecutive pair. This is exactly what "
            "eval_func compares against [lower_bounds, upper_bounds]."
        ),
    },
    {
        "name": "is_feasible_path",
        "input": "path: list[int]",
        "output": "(bool, str | None)",
        "purpose": (
            "(True, None) if `path` starts at 1, ends at n, every consecutive "
            "pair has an arc, and every resource total lies in its bounds; else "
            "(False, error_message). Cheaper than tools['is_feasible'] when you "
            "already have a list-of-vertices path."
        ),
    },
    # ----- Construction -----
    {
        "name": "dijkstra_pure_length",
        "input": "(no args)",
        "output": "list[int] | None",
        "purpose": (
            "Plain Dijkstra from source=1 to target=n, IGNORING resource "
            "constraints. Returns the minimum-length path as a list of "
            "vertices, or None if unreachable. Its length is a LOWER BOUND on "
            "any feasible RCSP solution -- use it for pruning or as a sanity "
            "check. The returned path itself may violate resource bounds."
        ),
    },
    {
        "name": "greedy_extend_path",
        "input": "time_limit_s: float = 2.0",
        "output": "list[int] | None",
        "purpose": (
            "Greedy forward path construction: at each step pick the cheapest "
            "outgoing arc whose resource consumption still keeps cumulative "
            "totals within the upper bounds. Returns a feasible-by-upper-bound "
            "path to n that also respects lower bounds, or None if it gets "
            "stuck or its final totals violate any lower bound. Cheap warm "
            "start, no optimality guarantee."
        ),
    },
    {
        "name": "label_setting_pareto",
        "input": "time_limit_s: float = 5.0, k_labels: int = 200",
        "output": "list[int] | None",
        "purpose": (
            "Pareto label-setting algorithm -- the classic exact method for "
            "RCSP. Expands labels (cost, resource-totals) in Dijkstra order, "
            "prunes by dominance and upper bounds, returns the cheapest path "
            "to n whose totals also satisfy the lower bounds. When `k_labels` "
            "is large enough to retain every non-dominated label, this is "
            "EXACT; smaller `k_labels` trades accuracy for speed. Returns "
            "None if no feasible label is reached within `time_limit_s`."
        ),
    },
    # ----- Exact -----
    {
        "name": "ilp_rcsp",
        "input": "time_limit_s: float = 30.0",
        "output": "list[int] | None",
        "purpose": (
            "Exact RCSP via python-mip / CBC (open-source). Binary arc "
            "variables with standard shortest-path flow conservation, vertex-"
            "visit indicators to charge vertex resources, and resource lower/"
            "upper bounds as linear constraints. Returns the optimal path "
            "(list of vertices) or None if infeasible / no solution within "
            "`time_limit_s`. Best for small-to-medium instances (a few "
            "hundred vertices, a few thousand arcs); use label_setting_pareto "
            "for larger graphs."
        ),
    },
]
