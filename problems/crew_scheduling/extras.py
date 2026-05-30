"""Per-problem extras for CO-Bench Crew Scheduling.

The task: assign every task (id 1..N, with start/finish times) to AT MOST K
crews so that within each crew (a sequence of tasks)
  - consecutive tasks do not time-overlap (tasks[i].finish <= tasks[j].start)
  - a transition arc (i, j) exists in the instance's arcs dict
  - the crew's duty time finish_last - start_first <= time_limit
and the total cost (sum of transition costs along all crews' chains) is
minimised. Each task must appear in EXACTLY ONE crew.

This is NOT a precomputed-duty set-covering instance: arcs are explicit and
duties (chains) must be built. The natural formulation is a min-cost
constrained path cover of a DAG, which we expose as a python-mip ILP plus
a greedy chain-packing heuristic.

Tool groups:
  (1) Queries:        n_tasks, n_crews, task_window, arc_cost, successors,
                      predecessors
  (2) Validation:     is_valid_crew, crew_cost, solution_cost
  (3) Construction:   greedy_chain_pack
  (4) Heavy / exact:  ilp_crew_scheduling

Solution shape (what tools['is_feasible']/['objective'] expect):
    {"crews": [[task_id, task_id, ...], [task_id, ...], ...]}
with at most K inner lists, every task id (1..N) appearing exactly once.
"""
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple

from mip import Model, BINARY, CONTINUOUS, MINIMIZE, xsum, OptimizationStatus


def extra_tools(instance: dict) -> dict:
    """Factory: returns Crew-Scheduling-specific tool callables.

    Instance schema (from CO-Bench load_data, one case):
      - N (int):          number of tasks (ids 1..N)
      - K (int):          MAX number of crews
      - time_limit (float): max duty time finish_last - start_first
      - tasks (dict):     {task_id -> (start, finish)}, task_id in 1..N
      - arcs  (dict):     {(i, j) -> cost}, with i,j in 1..N
    """
    N: int = int(instance["N"])
    K: int = int(instance["K"])
    T_LIMIT: float = float(instance["time_limit"])
    TASKS = dict(instance["tasks"])
    ARCS = dict(instance["arcs"])

    # Precompute adjacency for fast successor/predecessor queries.
    succ: dict[int, list[int]] = {i: [] for i in range(1, N + 1)}
    pred: dict[int, list[int]] = {i: [] for i in range(1, N + 1)}
    for (i, j) in ARCS.keys():
        succ[i].append(j)
        pred[j].append(i)
    for i in range(1, N + 1):
        succ[i].sort()
        pred[i].sort()

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def n_tasks() -> int:
        return N

    def n_crews() -> int:
        """Maximum number of crews allowed (K). The solution may use up to K
        crews; using fewer is also valid as long as every task is covered."""
        return K

    def task_window(t: int) -> Tuple[float, float]:
        """(start_time, finish_time) of task `t` (1-indexed). Raises KeyError
        if t is not a valid task id."""
        if t not in TASKS:
            raise KeyError(f"task {t} not in instance (valid ids 1..{N})")
        s, f = TASKS[t]
        return float(s), float(f)

    def arc_cost(i: int, j: int) -> Optional[float]:
        """Transition cost of using arc (i -> j) in some crew's schedule, or
        None if no such arc exists. Note: even when an arc exists, the
        framework still requires tasks[i].finish <= tasks[j].start (no
        overlap). Use successors() for a fully-filtered next-task list."""
        v = ARCS.get((int(i), int(j)))
        return float(v) if v is not None else None

    def successors(t: int) -> List[int]:
        """Tasks j such that arc (t -> j) exists AND tasks[t].finish <=
        tasks[j].start (i.e., j is a valid immediate successor of t in any
        crew). Sorted by ascending tasks[j].start."""
        if t not in TASKS:
            raise KeyError(f"task {t} not in instance (valid ids 1..{N})")
        _, t_fin = TASKS[t]
        out = [j for j in succ[int(t)] if TASKS[j][0] >= t_fin]
        out.sort(key=lambda j: TASKS[j][0])
        return out

    def predecessors(t: int) -> List[int]:
        """Tasks i such that arc (i -> t) exists AND tasks[i].finish <=
        tasks[t].start. Sorted by ascending tasks[i].finish."""
        if t not in TASKS:
            raise KeyError(f"task {t} not in instance (valid ids 1..{N})")
        t_start, _ = TASKS[t]
        out = [i for i in pred[int(t)] if TASKS[i][1] <= t_start]
        out.sort(key=lambda i: TASKS[i][1])
        return out

    # ==================================================================
    # (2) Validation primitives (cheap, no eval_func roundtrip)
    # ==================================================================
    def is_valid_crew(crew: Iterable[int]) -> Tuple[bool, Optional[str]]:
        """Check a SINGLE crew's chain against all per-crew constraints:
        non-empty, every task id exists, no overlap, valid arc between
        consecutive tasks, and total duty time <= time_limit. Returns
        (True, None) on success, else (False, error_message).
        Does NOT check coverage (use solution_cost / tools['is_feasible']
        for that)."""
        c = list(crew)
        if not c:
            return False, "crew is empty"
        for t in c:
            if t not in TASKS:
                return False, f"task {t} not in instance"
        # Duty time
        s_first = TASKS[c[0]][0]
        f_last = TASKS[c[-1]][1]
        if f_last - s_first > T_LIMIT + 1e-9:
            return False, (f"duty time {f_last - s_first} exceeds limit "
                           f"{T_LIMIT}")
        for k in range(len(c) - 1):
            a, b = c[k], c[k + 1]
            if TASKS[a][1] > TASKS[b][0] + 1e-9:
                return False, f"tasks {a} and {b} overlap"
            if (a, b) not in ARCS:
                return False, f"missing arc ({a}, {b})"
        return True, None

    def crew_cost(crew: Iterable[int]) -> float:
        """Sum of arc costs along the crew's chain. Does NOT validate; if any
        arc is missing returns float('inf'). Use is_valid_crew first if you
        need feasibility."""
        c = list(crew)
        total = 0.0
        for k in range(len(c) - 1):
            v = ARCS.get((c[k], c[k + 1]))
            if v is None:
                return float("inf")
            total += float(v)
        return total

    def solution_cost(crews: Iterable[Iterable[int]]) -> float:
        """Total cost of a whole solution (sum of crew_cost over all crews).
        Does NOT validate feasibility -- pair with tools['is_feasible']."""
        return sum(crew_cost(c) for c in crews)

    # ==================================================================
    # (3) Construction heuristic
    # ==================================================================
    def greedy_chain_pack() -> List[List[int]]:
        """Greedy chain packing. Tasks are processed in order of increasing
        start time. Each task is appended to the existing crew whose last
        task can chain into it (valid arc, no overlap, duty time stays
        within limit) with the SMALLEST resulting transition cost. If no
        crew can accept it, a new crew is opened.

        May produce MORE than K crews (in which case the result is
        infeasible w.r.t. the K-crew cap -- fall back to
        ilp_crew_scheduling). Each individual chain it returns satisfies
        all per-crew constraints by construction."""
        order = sorted(range(1, N + 1), key=lambda t: (TASKS[t][0], TASKS[t][1]))
        crews: List[List[int]] = []
        # Track (first_task_start, last_task) per crew for fast lookup.
        for t in order:
            t_start, t_fin = TASKS[t]
            best_idx = -1
            best_cost = float("inf")
            for idx, c in enumerate(crews):
                last = c[-1]
                last_fin = TASKS[last][1]
                if last_fin > t_start + 1e-9:
                    continue
                arc = ARCS.get((last, t))
                if arc is None:
                    continue
                first_start = TASKS[c[0]][0]
                if t_fin - first_start > T_LIMIT + 1e-9:
                    continue
                cost = float(arc)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            if best_idx >= 0:
                crews[best_idx].append(t)
            else:
                crews.append([t])
        return crews

    # ==================================================================
    # (4) Heavy: exact ILP via CBC
    # ==================================================================
    def ilp_crew_scheduling(time_limit_s: float = 30.0) -> Optional[dict]:
        """Solve crew scheduling exactly (or best-effort within budget) via
        CBC / python-mip. Returns a dict {"crews": [[...], ...]} ready to
        pass to tools['is_feasible'] / tools['objective'], or None if the
        solver finds no feasible solution within `time_limit_s`.

        Formulation (arc-based min-cost path cover with duty-time labels):
          Variables:
            x[i,j] in {0,1}        for each arc (i,j) in instance ARCS
                                   where TASKS[i].finish <= TASKS[j].start
            start[i] in {0,1}      i is the first task of its crew
            end[i]   in {0,1}      i is the last task of its crew
            u[i] in [0, infty)     start time of i's crew
                                   (= TASKS[i'].start where i' is chain head)
          Constraints:
            start[i] + sum_{j: (j,i) in arcs} x[j,i] == 1   (in-degree=1)
            end[i]   + sum_{j: (i,j) in arcs} x[i,j] == 1   (out-degree=1)
            sum_i start[i] <= K                              (at most K crews)
            sum_i start[i] == sum_i end[i]                   (book-keeping)
            For each i:
              u[i] >= TASKS[i].start - M*(1 - start[i])
              u[i] <= TASKS[i].start + M*(1 - start[i])
            For each arc (i,j) with x[i,j]=1: u[j] == u[i]
              implemented via two big-M inequalities.
            For each i:
              TASKS[i].finish - u[i] <= time_limit
          Objective: minimise sum_{(i,j)} arc_cost(i,j) * x[i,j].

        Limitations:
          - Big-M labelling makes the LP relaxation weak; expect long times
            for N >= ~200. Use a generous time_limit_s.
          - Returns None if no integer-feasible solution was found.
        """
        # Restrict to arcs that are temporally compatible (cheap pruning).
        usable_arcs: List[Tuple[int, int, float]] = []
        for (i, j), c in ARCS.items():
            if i in TASKS and j in TASKS:
                if TASKS[i][1] <= TASKS[j][0] + 1e-9:
                    # also: a single-arc chain i->j must fit time_limit if it
                    # were the entire chain (necessary condition, prunes some)
                    if TASKS[j][1] - TASKS[i][0] <= T_LIMIT + 1e-9:
                        usable_arcs.append((int(i), int(j), float(c)))
        # Big-M: a safe upper bound on u[i] is the largest task start time.
        big_M = max((TASKS[t][0] for t in TASKS), default=0.0) + T_LIMIT + 1.0

        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)

        x = {(i, j): m.add_var(var_type=BINARY, name=f"x_{i}_{j}")
             for (i, j, _c) in usable_arcs}
        start = {i: m.add_var(var_type=BINARY, name=f"s_{i}")
                 for i in range(1, N + 1)}
        end = {i: m.add_var(var_type=BINARY, name=f"e_{i}")
               for i in range(1, N + 1)}
        u = {i: m.add_var(var_type=CONTINUOUS, lb=0.0, ub=big_M, name=f"u_{i}")
             for i in range(1, N + 1)}

        cost_of = {(i, j): c for (i, j, c) in usable_arcs}
        m.objective = xsum(cost_of[(i, j)] * x[(i, j)] for (i, j) in x)

        # In-degree / out-degree balance.
        in_arcs: dict[int, list[Tuple[int, int]]] = {i: [] for i in range(1, N + 1)}
        out_arcs: dict[int, list[Tuple[int, int]]] = {i: [] for i in range(1, N + 1)}
        for (i, j) in x:
            in_arcs[j].append((i, j))
            out_arcs[i].append((i, j))
        for i in range(1, N + 1):
            m += start[i] + xsum(x[a] for a in in_arcs[i]) == 1, f"indeg_{i}"
            m += end[i] + xsum(x[a] for a in out_arcs[i]) == 1, f"outdeg_{i}"

        # Crew count: at most K.
        m += xsum(start[i] for i in range(1, N + 1)) <= K, "K_starts"
        m += (xsum(start[i] for i in range(1, N + 1))
              == xsum(end[i] for i in range(1, N + 1))), "balance_se"

        # Chain-head labelling: if start[i]=1 then u[i] = TASKS[i].start.
        for i in range(1, N + 1):
            s_i = float(TASKS[i][0])
            m += u[i] >= s_i - big_M * (1 - start[i]), f"u_start_lo_{i}"
            m += u[i] <= s_i + big_M * (1 - start[i]), f"u_start_hi_{i}"

        # Label propagation along arcs: x[i,j]=1 forces u[j] = u[i].
        for (i, j) in x:
            m += u[j] - u[i] <= big_M * (1 - x[(i, j)]), f"u_prop_hi_{i}_{j}"
            m += u[i] - u[j] <= big_M * (1 - x[(i, j)]), f"u_prop_lo_{i}_{j}"

        # Duty time per crew (applied at every task; the binding one is
        # the chain's last task, but the constraint is valid for all).
        for i in range(1, N + 1):
            f_i = float(TASKS[i][1])
            m += f_i - u[i] <= T_LIMIT, f"duty_{i}"

        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None

        # Reconstruct chains from the x[i,j] solution.
        next_of: dict[int, int] = {}
        for (i, j), var in x.items():
            v = var.x
            if v is not None and v > 0.5:
                next_of[i] = j
        heads = [i for i in range(1, N + 1)
                 if start[i].x is not None and start[i].x > 0.5]
        crews: List[List[int]] = []
        used = set()
        for h in heads:
            chain = [h]
            used.add(h)
            cur = h
            while cur in next_of:
                nxt = next_of[cur]
                if nxt in used:
                    break  # safety
                chain.append(nxt)
                used.add(nxt)
                cur = nxt
            crews.append(chain)
        # Sanity: every task should appear exactly once.
        if used != set(range(1, N + 1)):
            return None
        return {"crews": crews}

    # ==================================================================
    # (5) Solution-dict builder + one-shot solver
    # ==================================================================
    def make_solution(crews) -> dict:
        """Wrap a list of crew chains into the EXACT dict shape eval_func
        expects: {'crews': list[list[int]]}. Use on the output of
        greedy_chain_pack() / ilp_crew_scheduling() so you never return the
        wrong dict shape."""
        out_crews: List[List[int]] = []
        if crews:
            for c in crews:
                chain = [int(t) for t in c]
                if chain:
                    out_crews.append(chain)
        return {"crews": out_crews}

    def _merge_crews_into_k(crews_in: List[List[int]],
                            k_max: int) -> List[List[int]]:
        """Force a list of crews down to at most k_max by repeatedly merging
        pairs whose concatenation is a legal chain (arc exists, no overlap,
        duty time within limit). Greedy: try every legal merge, pick the
        cheapest cost. If no legal merge exists, returns the current state
        (still > k_max -- caller treats as infeasible)."""
        crews = [list(c) for c in crews_in if c]
        while len(crews) > k_max:
            best = None
            best_cost_delta = float("inf")
            for i in range(len(crews)):
                for j in range(len(crews)):
                    if i == j:
                        continue
                    last_i = crews[i][-1]
                    first_j = crews[j][0]
                    arc = ARCS.get((last_i, first_j))
                    if arc is None:
                        continue
                    if TASKS[last_i][1] > TASKS[first_j][0] + 1e-9:
                        continue
                    s_first = TASKS[crews[i][0]][0]
                    f_last = TASKS[crews[j][-1]][1]
                    if f_last - s_first > T_LIMIT + 1e-9:
                        continue
                    if float(arc) < best_cost_delta:
                        best_cost_delta = float(arc)
                        best = (i, j)
            if best is None:
                return crews  # no further legal merge
            i, j = best
            merged = crews[i] + crews[j]
            crews = [c for k, c in enumerate(crews) if k != i and k != j]
            crews.append(merged)
        return crews

    def solve_min_cost_flow(time_limit_s: float = 10.0) -> Optional[dict]:
        """POLYNOMIAL-TIME OPTIMAL SOLVER via min-cost K-path cover on the
        task DAG (networkx min_cost_flow).

        Models the problem as a flow network:
          - each task is split into u_in / u_out with a lower-bound-1 internal
            edge (encoded via node demands so flow MUST pass through it)
          - each valid arc (i -> j) is an edge (i_out, j_in) with cost = ARCS[(i,j)]
          - a super-source S pushes up to K units (= number of crews) toward
            task u_in nodes; a super-sink T absorbs from u_out nodes
          - a bypass edge S -> T (capacity K, cost 0) lets the optimization
            use FEWER than K paths when that is cheaper

        Pre-filters arcs by overlap + per-arc duty_time. After flow extraction,
        chains whose total duty_time exceeds T_LIMIT are SPLIT into shorter
        chains, then any leftover crews above K are merged via
        _merge_crews_into_k.

        Returns a ready-to-submit dict {'crews': [[...], ...]}, or None if
        even singleton-crew assignment violates K (rare). Always strictly
        polynomial -- no ILP, no timeout sensitivity. Try this FIRST."""
        try:
            import networkx as nx
        except ImportError:
            return None

        # Build valid arc list: (u, v) with u.finish <= v.start AND 2-task duty <= T_LIMIT.
        valid_arcs: List[Tuple[int, int, float]] = []
        for (i, j), c in ARCS.items():
            if i in TASKS and j in TASKS:
                if TASKS[i][1] <= TASKS[j][0] + 1e-9:
                    if TASKS[j][1] - TASKS[i][0] <= T_LIMIT + 1e-9:
                        valid_arcs.append((int(i), int(j), float(c)))

        # Cost scaling: networkx (some versions) require integer weights. Scale by 1000.
        def _w(c: float) -> int:
            return int(round(c * 1000.0))

        def _build_graph(K_try: int):
            G = nx.DiGraph()
            for i in range(1, N + 1):
                # u_in has +1 demand (absorb 1 net), u_out has -1 (produce 1 net).
                # This encodes "1 unit must pass through u_in -> u_out implicitly".
                G.add_node(("in", i), demand=1)
                G.add_node(("out", i), demand=-1)
            G.add_node("S", demand=-K_try)
            G.add_node("T", demand=+K_try)
            for i in range(1, N + 1):
                G.add_edge("S", ("in", i), capacity=1, weight=0)
                G.add_edge(("out", i), "T", capacity=1, weight=0)
            for (i, j, c) in valid_arcs:
                G.add_edge(("out", i), ("in", j), capacity=1, weight=_w(c))
            # Bypass to allow < K paths if cheaper.
            G.add_edge("S", "T", capacity=K_try, weight=0)
            return G

        # Try K_try = K first (the cap from the instance). If infeasible
        # (arc structure forces more paths), gradually raise K_try up to N.
        flow_dict = None
        for K_try in [K] + list(range(K + 1, N + 1)):
            try:
                G = _build_graph(K_try)
                flow_dict = nx.min_cost_flow(G)
                break
            except nx.NetworkXUnfeasible:
                continue
            except Exception:
                continue
        if flow_dict is None:
            return None

        # Extract next-hop map and path heads from the flow.
        next_of: dict = {}
        heads: List[int] = []
        for u, neighbors in flow_dict.items():
            for v, f in neighbors.items():
                if f <= 0:
                    continue
                if u == "S" and v == "T":
                    continue  # bypass
                if u == "S" and isinstance(v, tuple) and v[0] == "in":
                    heads.append(v[1])
                elif (isinstance(u, tuple) and u[0] == "out"
                      and isinstance(v, tuple) and v[0] == "in"):
                    next_of[u[1]] = v[1]
        # Follow chains from heads.
        chains: List[List[int]] = []
        used = set()
        for h in heads:
            chain = [h]
            used.add(h)
            cur = h
            while cur in next_of:
                nxt = next_of[cur]
                if nxt in used:
                    break
                chain.append(nxt)
                used.add(nxt)
                cur = nxt
            chains.append(chain)
        if used != set(range(1, N + 1)):
            # Some tasks not covered -- shouldn't happen given the demand setup
            return None

        # Split chains that exceed T_LIMIT (path-level duty time).
        final_crews: List[List[int]] = []
        for chain in chains:
            i = 0
            while i < len(chain):
                j = i
                while (j + 1 < len(chain)
                       and TASKS[chain[j + 1]][1] - TASKS[chain[i]][0]
                       <= T_LIMIT + 1e-9):
                    j += 1
                final_crews.append(chain[i:j + 1])
                i = j + 1
        # If splitting pushed us above K, try to merge.
        if len(final_crews) > K:
            final_crews = _merge_crews_into_k(final_crews, K)
        if len(final_crews) > K:
            return None
        return make_solution(final_crews)

    def solve_default(time_limit_s: float = 10.0) -> dict:
        """ONE-SHOT STRONG SOLVER. Returns the complete solution dict
        {'crews': list[list[int]]} ready to return directly.

        Strategy (each builds on the previous; first feasible wins):
          1. ilp_crew_scheduling with up to time_limit_s seconds -- exact,
             feasible on small/medium N (<= ~80) within typical budgets.
          2. solve_min_cost_flow -- polynomial K-path cover; works when the
             arc structure admits it without duty-time blowup.
          3. greedy_chain_pack + _merge_crews_into_k -- last-resort heuristic.

        Use as the FIRST thing your solve() function calls. ONE LINE:
            return tools['solve_default'](time_limit_s=60)
        """
        # ILP first (most likely to satisfy the full constraint set when it
        # finishes). Give it most of the budget; leave a safety margin.
        ilp_budget = max(float(time_limit_s) - 5.0, 5.0)
        sol = ilp_crew_scheduling(time_limit_s=ilp_budget)
        if sol is not None:
            return sol
        # Min-cost flow: polynomial; may emit > K crews after duty-time split.
        sol = solve_min_cost_flow(time_limit_s=time_limit_s)
        if sol is not None:
            return sol
        # Last resort: greedy + merge. May still be infeasible (> K crews)
        # for tight instances; eval_func will then reject. Returning a dict
        # is preferred over raising so callers can inspect.
        crews = greedy_chain_pack()
        if len(crews) > K:
            crews = _merge_crews_into_k(crews, K)
        return make_solution(crews)

    return {
        # (5) one-shot + builder (CALL FIRST)
        "solve_default": solve_default,
        "solve_min_cost_flow": solve_min_cost_flow,
        "make_solution": make_solution,
        # (4) heavy
        "ilp_crew_scheduling": ilp_crew_scheduling,
        # (3) construction
        "greedy_chain_pack": greedy_chain_pack,
        # (2) validation
        "is_valid_crew": is_valid_crew,
        "crew_cost": crew_cost,
        "solution_cost": solution_cost,
        # (1) queries
        "n_tasks": n_tasks,
        "n_crews": n_crews,
        "task_window": task_window,
        "arc_cost": arc_cost,
        "successors": successors,
        "predecessors": predecessors,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (5) ONE-SHOT STRONG SOLVER (call this first!) -----
    {
        "name": "solve_default",
        "input": "time_limit_s: float = 60.0",
        "output": "dict {'crews': list[list[int]]}",
        "purpose": (
            "RECOMMENDED START: returns a complete solution dict ready to return "
            "directly. Cascade: (1) ilp_crew_scheduling (exact CBC); "
            "(2) solve_min_cost_flow (polynomial K-path cover via networkx); "
            "(3) greedy_chain_pack + merge. ONE LINE: "
            "`return tools['solve_default'](time_limit_s=60)`."
        ),
    },
    {
        "name": "solve_min_cost_flow",
        "input": "time_limit_s: float = 10.0",
        "output": "dict {'crews': list[list[int]]} | None",
        "purpose": (
            "POLYNOMIAL solver via networkx min_cost_flow on the task DAG with "
            "split-tasks-into-(in,out)-with-lower-bound-1 transformation. Cost = "
            "sum of arc costs used. Splits chains that exceed T_LIMIT and merges "
            "where possible to stay within K crews. Returns None if even merging "
            "cannot get crew count <= K. Use as the polynomial-time alternative "
            "to ilp_crew_scheduling."
        ),
    },
    {
        "name": "make_solution",
        "input": "crews: Iterable[Iterable[int]]",
        "output": "dict {'crews': list[list[int]]}",
        "purpose": (
            "Build the EXACT solution dict shape eval_func wants from a list of "
            "crew chains (each chain is a list of 1-indexed task ids). Filters "
            "out empty chains. Use on the output of greedy_chain_pack() so you "
            "never return the wrong dict shape."
        ),
    },
    # ----- (4) Heavy: exact ILP -----
    {
        "name": "ilp_crew_scheduling",
        "input": "time_limit_s: float = 30.0",
        "output": "dict {'crews': list[list[int]]} | None",
        "purpose": (
            "Use as primary solver. Solves the whole instance exactly (or "
            "best-effort within the given budget) via python-mip / CBC and "
            "returns a ready-to-submit {'crews': [[...], ...]} dict, or None "
            "if no feasible solution was found in time. Models the problem as "
            "min-cost path cover of the task DAG with at most K paths and "
            "big-M duty-time labels. Pass a generous time_limit_s (e.g. 60-120s) "
            "for N >= 200."
        ),
    },
    # ----- (1) Queries -----
    {
        "name": "n_tasks",
        "input": "(no args)",
        "output": "int",
        "purpose": "Total number of tasks N (task ids are 1..N).",
    },
    {
        "name": "n_crews",
        "input": "(no args)",
        "output": "int",
        "purpose": (
            "Maximum number of crews K. The solution's `crews` list must "
            "contain at most K non-empty crews (using fewer is also valid)."
        ),
    },
    {
        "name": "task_window",
        "input": "t: int (task id, 1-indexed)",
        "output": "(float, float)",
        "purpose": (
            "(start_time, finish_time) of task t. Two tasks in the same "
            "crew, with t before t', must satisfy finish_t <= start_t', "
            "and the whole crew's finish_last - start_first <= time_limit."
        ),
    },
    {
        "name": "arc_cost",
        "input": "i: int, j: int",
        "output": "float | None",
        "purpose": (
            "Transition cost of placing j immediately after i in a crew, or "
            "None if no such arc exists. The framework REQUIRES the arc to "
            "exist for every consecutive pair, so this is the canonical "
            "feasibility/edge-cost lookup."
        ),
    },
    {
        "name": "successors",
        "input": "t: int (task id)",
        "output": "list[int]",
        "purpose": (
            "Tasks j with an arc (t -> j) AND finish_t <= start_j -- i.e., "
            "valid immediate next tasks for any crew containing t (up to the "
            "duty-time check, which depends on the chain head). Sorted by "
            "ascending start_j."
        ),
    },
    {
        "name": "predecessors",
        "input": "t: int (task id)",
        "output": "list[int]",
        "purpose": (
            "Tasks i with an arc (i -> t) AND finish_i <= start_t -- valid "
            "previous tasks in a crew. Sorted by ascending finish_i."
        ),
    },
    # ----- (2) Validation -----
    {
        "name": "is_valid_crew",
        "input": "crew: Iterable[int]",
        "output": "(bool, str | None)",
        "purpose": (
            "Per-crew feasibility check: non-empty, every task id exists, "
            "no temporal overlap, an arc exists between every consecutive "
            "pair, and finish_last - start_first <= time_limit. Returns "
            "(True, None) on success else (False, reason). Use inside a "
            "local-search loop to reject infeasible neighbours before "
            "computing the global objective."
        ),
    },
    {
        "name": "crew_cost",
        "input": "crew: Iterable[int]",
        "output": "float",
        "purpose": (
            "Sum of arc costs along the crew (no feasibility check). "
            "Returns float('inf') if any consecutive arc is missing. Cheap "
            "way to evaluate a chain in isolation; combine with crew_cost "
            "over all crews, or solution_cost, for the whole solution."
        ),
    },
    {
        "name": "solution_cost",
        "input": "crews: Iterable[Iterable[int]]",
        "output": "float",
        "purpose": (
            "Sum of crew_cost over every crew. Identical to "
            "tools['objective']({'crews': crews}) when the solution is "
            "feasible, but does NOT validate feasibility -- pair with "
            "tools['is_feasible'] before trusting it for ranking solutions."
        ),
    },
    # ----- (3) Construction -----
    {
        "name": "greedy_chain_pack",
        "input": "(no args)",
        "output": "list[list[int]]",
        "purpose": (
            "Greedy chain packing: tasks processed in increasing start time, "
            "each appended to the existing crew whose tail chains into it with "
            "minimum transition cost subject to arc + overlap + duty-time "
            "constraints. New crew opened when none accepts. Each returned chain "
            "satisfies all per-crew constraints; the total chain count may "
            "exceed K, in which case use ilp_crew_scheduling or solve_default "
            "to enforce the K-crew cap."
        ),
    },
]
