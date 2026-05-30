"""Per-problem extras for Capacitated Warehouse Location (CWL).

Provides primitive building blocks so the LLM can compose construction /
repair / LNS heuristics for CWL without re-deriving the ILP, capacity
bookkeeping, or basic neighborhood moves.

Tool groups:
  (1) Queries:        warehouse_fixed_cost, warehouse_capacity, customer_demand,
                      serve_cost, n_warehouses, n_customers
  (2) Feasibility:    warehouse_load, warehouse_remaining, unassigned_customers,
                      total_cost
  (3) Construction /
      improvement:    greedy_open_by_density, greedy_serve_nearest,
                      apply_reassign_customer, apply_swap_open_close,
                      to_solution
  (4) Exact / heavy:  ilp_cwl

CO-Bench requires a solution dict with three keys:
  - 'total_cost':      float (objective value)
  - 'warehouse_open':  list[int] of length m, entries in {0, 1}
  - 'assignments':     list[list[float]] of shape n x m, where
                       assignments[j][i] is the amount of customer j's demand
                       supplied by warehouse i (splittable demand).

The CWL benchmark (BB81 cap*.txt) admits single-source optimal solutions, so
most tools here use a SINGLE-SOURCE assignment representation:
  - `assignment` is a list[int] of length n, with `assignment[j] = i` meaning
    customer j is served (entirely) by warehouse i (0-indexed).
  - `open_set` is a Python set / iterable of 0-indexed warehouse ids that are
    currently OPEN.

Use `to_solution(open_set, assignment)` to convert the (open_set, assignment)
pair into the CO-Bench dict before returning it from `solve`.
"""
from __future__ import annotations
import time
from typing import Optional, Iterable, List, Set

from mip import Model, BINARY, CONTINUOUS, MINIMIZE, xsum, OptimizationStatus


def extra_tools(instance: dict) -> dict:
    """Factory: returns CWL-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench CWL load_data, one case):
      - m:           int, number of candidate warehouses
      - n:           int, number of customers
      - warehouses:  list[dict] of length m with keys 'capacity', 'fixed_cost'
      - customers:   list[dict] of length n with keys 'demand', 'costs'
                     (costs is a list[float] of length m: per-unit cost from
                     each warehouse).
    """
    m: int = int(instance["m"])
    n: int = int(instance["n"])
    warehouses = instance["warehouses"]
    customers = instance["customers"]

    F = [float(w["fixed_cost"]) for w in warehouses]
    Q = [float(w["capacity"]) for w in warehouses]
    D = [float(c["demand"]) for c in customers]
    # serve_cost[i][j] = per-unit cost * demand[j]  (= cost to fully serve j from i)
    # Note: costs are stored as customers[j]['costs'][i] (per-unit), so total
    # cost when warehouse i fully serves customer j is costs[j][i] (the
    # weighted-average eval reduces to costs[j][i] when fraction == 1).
    C = [[float(customers[j]["costs"][i]) for j in range(n)] for i in range(m)]

    def _i(i: int) -> int:
        i = int(i)
        if not (0 <= i < m):
            raise ValueError(f"warehouse id {i} out of range [0, {m})")
        return i

    def _j(j: int) -> int:
        j = int(j)
        if not (0 <= j < n):
            raise ValueError(f"customer id {j} out of range [0, {n})")
        return j

    def _validate_assignment(assignment: Iterable[int]) -> List[int]:
        a = list(assignment)
        if len(a) != n:
            raise ValueError(f"assignment has length {len(a)}, expected {n}")
        return a

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def warehouse_fixed_cost(i: int) -> float:
        return F[_i(i)]

    def warehouse_capacity(i: int) -> float:
        return Q[_i(i)]

    def customer_demand(j: int) -> float:
        return D[_j(j)]

    def serve_cost(i: int, j: int) -> float:
        """Total cost (NOT per-unit) of customer j being served fully by
        warehouse i. Equivalent to customers[j]['costs'][i] in the eval_func
        convention (weighted-average over a single warehouse is just that
        warehouse's cost)."""
        return C[_i(i)][_j(j)]

    def n_warehouses() -> int:
        return m

    def n_customers() -> int:
        return n

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def warehouse_load(i: int, assignment: Iterable[int]) -> float:
        """Total demand assigned to warehouse i under a single-source
        assignment. Entries in `assignment` outside [0, m) are ignored."""
        i0 = _i(i)
        a = _validate_assignment(assignment)
        total = 0.0
        for j, aj in enumerate(a):
            try:
                if int(aj) == i0:
                    total += D[j]
            except (TypeError, ValueError):
                continue
        return total

    def warehouse_remaining(i: int, assignment: Iterable[int]) -> float:
        """capacity[i] - warehouse_load(i, assignment). Can be negative if
        the assignment overuses warehouse i."""
        return Q[_i(i)] - warehouse_load(i, assignment)

    def unassigned_customers(assignment: Iterable[int]) -> List[int]:
        """List of customer indices whose `assignment[j]` is missing or
        invalid (None, negative, or >= m). Useful during incremental
        construction."""
        a = _validate_assignment(assignment)
        out = []
        for j, aj in enumerate(a):
            if aj is None:
                out.append(j)
                continue
            try:
                v = int(aj)
            except (TypeError, ValueError):
                out.append(j)
                continue
            if v < 0 or v >= m:
                out.append(j)
        return out

    def total_cost(open_set: Iterable[int],
                   assignment: Iterable[int]) -> float:
        """Total cost of a (open_set, single-source assignment) pair:
        sum of fixed costs over open warehouses plus sum of serve_cost(i,j)
        over all customers. Does NOT check feasibility -- use is_feasible
        on tools['is_feasible'](to_solution(...)) for that."""
        a = _validate_assignment(assignment)
        opens = {int(i) for i in open_set if 0 <= int(i) < m}
        cost = sum(F[i] for i in opens)
        for j, aj in enumerate(a):
            try:
                i0 = int(aj)
            except (TypeError, ValueError):
                continue
            if 0 <= i0 < m:
                cost += C[i0][j]
        return float(cost)

    # ==================================================================
    # (3) Construction / improvement
    # ==================================================================
    def greedy_open_by_density() -> tuple:
        """Open warehouses greedily by the smallest fixed_cost / capacity
        ratio (cheapest 'cost per unit of capacity') until cumulative
        capacity >= total demand. Then assign each customer to the nearest
        feasible open warehouse via greedy_serve_nearest.

        Returns (open_set, assignment) where:
          - open_set is a sorted list[int] of 0-indexed warehouse ids
          - assignment is a list[int] of length n (single-source)
        If no feasible assignment is found, customers may be left with
        assignment[j] = -1; the caller should check via
        unassigned_customers and decide how to repair (e.g., open more
        warehouses or call ilp_cwl)."""
        total_demand = sum(D)
        # density = fixed_cost / capacity. Smaller is better.
        order = sorted(range(m), key=lambda i: F[i] / Q[i] if Q[i] > 0
                       else float("inf"))
        opens: List[int] = []
        cap_so_far = 0.0
        for i in order:
            opens.append(i)
            cap_so_far += Q[i]
            if cap_so_far >= total_demand:
                break
        opens.sort()
        assignment = greedy_serve_nearest(opens)
        return opens, assignment

    def greedy_serve_nearest(open_set: Iterable[int]) -> List[int]:
        """Assign each customer to the cheapest open warehouse with enough
        remaining capacity. Customers are processed in decreasing-demand
        order (hardest first). If no open warehouse can fit a customer, the
        slot is set to -1. Returns a length-n list[int] of 0-indexed
        warehouse ids."""
        opens = sorted({int(i) for i in open_set if 0 <= int(i) < m})
        load = {i: 0.0 for i in opens}
        assignment = [-1] * n
        # Hardest (largest demand) customers first.
        cust_order = sorted(range(n), key=lambda j: -D[j])
        for j in cust_order:
            best_i = -1
            best_c = float("inf")
            for i in opens:
                if load[i] + D[j] <= Q[i] + 1e-9:
                    if C[i][j] < best_c:
                        best_c = C[i][j]
                        best_i = i
            if best_i >= 0:
                assignment[j] = best_i
                load[best_i] += D[j]
            # else leave as -1 (unassigned)
        return assignment

    def apply_reassign_customer(assignment: Iterable[int], customer: int,
                                new_warehouse: int) -> Optional[List[int]]:
        """Return a NEW assignment with customer `customer` moved to
        `new_warehouse` (0-indexed). Returns None if the move would exceed
        `new_warehouse`'s capacity given the rest of the assignment. The
        input list is never mutated. Does NOT check that new_warehouse is
        open -- the caller is responsible for that."""
        a = _validate_assignment(assignment)
        j = _j(customer)
        new_i = _i(new_warehouse)
        # Recompute load on new_i excluding customer j.
        new_load = 0.0
        for jj, aj in enumerate(a):
            if jj == j:
                continue
            try:
                if int(aj) == new_i:
                    new_load += D[jj]
            except (TypeError, ValueError):
                continue
        new_load += D[j]
        if new_load > Q[new_i] + 1e-9:
            return None
        out = list(a)
        out[j] = new_i
        return out

    def apply_swap_open_close(open_set: Iterable[int],
                              time_limit_s: float = 2.0) -> tuple:
        """Local search over the OPEN/CLOSE status of warehouses, holding
        customer routing implicit (re-derived via greedy_serve_nearest after
        each candidate move). Tries:
          - opening one currently-closed warehouse, or
          - closing one currently-open warehouse, or
          - swapping one open <-> one closed
        First-improvement, until time_limit_s is exhausted or no move
        improves. Returns (open_set, assignment) as a tuple where open_set
        is a sorted list[int] of 0-indexed ids."""
        t0 = time.time()
        safety = 0.05
        opens = sorted({int(i) for i in open_set if 0 <= int(i) < m})

        def _score(opens_list):
            assn = greedy_serve_nearest(opens_list)
            # Penalize unassigned customers heavily so they are avoided.
            if any(a < 0 for a in assn):
                return float("inf"), assn
            return total_cost(opens_list, assn), assn

        best_cost, best_assn = _score(opens)
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            closed = [i for i in range(m) if i not in opens]
            # 1) Try opening one closed warehouse.
            for k in closed:
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                cand = sorted(opens + [k])
                c, a = _score(cand)
                if c < best_cost - 1e-9:
                    best_cost, best_assn = c, a
                    opens = cand
                    improved = True
                    break
            if improved:
                continue
            # 2) Try closing one open warehouse (only if it doesn't strand demand).
            for k in list(opens):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                cand = [i for i in opens if i != k]
                if not cand:
                    continue
                if sum(Q[i] for i in cand) < sum(D) - 1e-9:
                    continue  # insufficient capacity
                c, a = _score(cand)
                if c < best_cost - 1e-9:
                    best_cost, best_assn = c, a
                    opens = cand
                    improved = True
                    break
            if improved:
                continue
            # 3) Swap: close one open and open one closed.
            for k_close in list(opens):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                for k_open in closed:
                    cand = sorted([i for i in opens if i != k_close] + [k_open])
                    if sum(Q[i] for i in cand) < sum(D) - 1e-9:
                        continue
                    c, a = _score(cand)
                    if c < best_cost - 1e-9:
                        best_cost, best_assn = c, a
                        opens = cand
                        improved = True
                        break
                if improved:
                    break
        return opens, best_assn

    def to_solution(open_set: Iterable[int],
                    assignment: Iterable[int]) -> dict:
        """Convert a (open_set, single-source assignment) pair into the
        CO-Bench solution dict with keys 'total_cost', 'warehouse_open',
        'assignments'. The returned 'assignments' is an n x m matrix where
        assignments[j][i] = D[j] iff assignment[j] == i, else 0.0."""
        a = _validate_assignment(assignment)
        opens = {int(i) for i in open_set if 0 <= int(i) < m}
        warehouse_open = [1 if i in opens else 0 for i in range(m)]
        assignments = [[0.0] * m for _ in range(n)]
        for j, aj in enumerate(a):
            try:
                i0 = int(aj)
            except (TypeError, ValueError):
                continue
            if 0 <= i0 < m:
                assignments[j][i0] = D[j]
        return {
            "total_cost": total_cost(opens, a),
            "warehouse_open": warehouse_open,
            "assignments": assignments,
        }

    # ==================================================================
    # (4) Exact / heavy: ILP
    # ==================================================================
    def ilp_cwl(time_limit_s: float = 10.0) -> Optional[dict]:
        """Solve CWL exactly via CBC (python-mip). Supports SPLITTABLE demand
        (a customer's demand may be served by multiple warehouses), which is
        required by some CO-Bench instances (e.g. cap41.txt has a customer
        with demand 5495 > any single warehouse's capacity 5000).

        Variables:
          - y[i] in {0,1}    : warehouse i is open
          - x[i,j] in [0, 1] : FRACTION of customer j's demand served by i

        Objective:
          minimize sum_i F[i]*y[i] + sum_{i,j} costs[j][i] * x[i,j]
        (CO-Bench cost per customer = weighted-average over warehouses of
        costs[j][i], so the LP objective equals it directly.)

        Constraints:
          - sum_i x[i,j] == 1                  for each customer j
          - sum_j D[j]*x[i,j] <= Q[i]*y[i]     for each warehouse i

        Returns a CO-Bench solution dict, or None if no feasible solution
        within budget."""
        model = Model(sense=MINIMIZE)
        model.verbose = 0
        model.max_seconds = float(time_limit_s)
        y = [model.add_var(var_type=BINARY, name=f"y_{i}") for i in range(m)]
        x = [[model.add_var(var_type=CONTINUOUS, lb=0.0, ub=1.0,
                            name=f"x_{i}_{j}")
              for j in range(n)] for i in range(m)]
        model.objective = (
            xsum(F[i] * y[i] for i in range(m))
            + xsum(C[i][j] * x[i][j] for i in range(m) for j in range(n))
        )
        for j in range(n):
            model += xsum(x[i][j] for i in range(m)) == 1, f"cust_{j}"
        for i in range(m):
            model += xsum(D[j] * x[i][j] for j in range(n)) <= Q[i] * y[i], f"cap_{i}"
        status = model.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if model.num_solutions < 1:
            return None
        opens: List[int] = []
        for i in range(m):
            val = y[i].x
            if val is not None and val > 0.5:
                opens.append(i)
        warehouse_open = [1 if i in set(opens) else 0 for i in range(m)]
        # Build the splittable assignments matrix directly: n x m, where
        # assignments[j][i] = allocation amount (= D[j] * fraction).
        assignments = [[0.0] * m for _ in range(n)]
        total_cost = sum(F[i] for i in opens)
        for j in range(n):
            for i in range(m):
                xv = x[i][j].x
                if xv is None:
                    continue
                if xv > 1e-9:
                    assignments[j][i] = float(D[j]) * float(xv)
                    total_cost += float(xv) * float(C[i][j])
        return {
            "total_cost": float(total_cost),
            "warehouse_open": warehouse_open,
            "assignments": assignments,
        }

    return {
        # (1) queries
        "warehouse_fixed_cost": warehouse_fixed_cost,
        "warehouse_capacity": warehouse_capacity,
        "customer_demand": customer_demand,
        "serve_cost": serve_cost,
        "n_warehouses": n_warehouses,
        "n_customers": n_customers,
        # (2) feasibility
        "warehouse_load": warehouse_load,
        "warehouse_remaining": warehouse_remaining,
        "unassigned_customers": unassigned_customers,
        "total_cost": total_cost,
        # (3) construction / improvement
        "greedy_open_by_density": greedy_open_by_density,
        "greedy_serve_nearest": greedy_serve_nearest,
        "apply_reassign_customer": apply_reassign_customer,
        "apply_swap_open_close": apply_swap_open_close,
        "to_solution": to_solution,
        # (4) exact
        "ilp_cwl": ilp_cwl,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "warehouse_fixed_cost",
        "input": "i: int (0-indexed warehouse)",
        "output": "float",
        "purpose": "Fixed opening cost of warehouse i (warehouses[i]['fixed_cost']).",
    },
    {
        "name": "warehouse_capacity",
        "input": "i: int (0-indexed warehouse)",
        "output": "float",
        "purpose": "Capacity of warehouse i (warehouses[i]['capacity']).",
    },
    {
        "name": "customer_demand",
        "input": "j: int (0-indexed customer)",
        "output": "float",
        "purpose": "Demand of customer j (customers[j]['demand']).",
    },
    {
        "name": "serve_cost",
        "input": "i: int (0-indexed warehouse), j: int (0-indexed customer)",
        "output": "float",
        "purpose": (
            "Total cost of warehouse i fully serving customer j -- equals "
            "customers[j]['costs'][i] in the CO-Bench convention (the "
            "weighted-average cost reduces to this value when the customer is "
            "served by a single warehouse)."
        ),
    },
    {
        "name": "n_warehouses",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of candidate warehouse sites m.",
    },
    {
        "name": "n_customers",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of customers n.",
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "warehouse_load",
        "input": "i: int (0-indexed), assignment: list[int]",
        "output": "float",
        "purpose": (
            "Total demand assigned to warehouse i under a SINGLE-SOURCE "
            "assignment list (entries are 0-indexed warehouse ids; entries "
            "outside [0, m) are treated as unassigned)."
        ),
    },
    {
        "name": "warehouse_remaining",
        "input": "i: int (0-indexed), assignment: list[int]",
        "output": "float",
        "purpose": (
            "capacity[i] - warehouse_load(i, assignment). Negative if "
            "warehouse i is over capacity under `assignment`."
        ),
    },
    {
        "name": "unassigned_customers",
        "input": "assignment: list[int]",
        "output": "list[int]",
        "purpose": (
            "0-indexed customer indices whose entry is missing/invalid (None, "
            "negative, or >= m). Use this to drive incremental construction "
            "and to detect cases where greedy left a customer unplaced."
        ),
    },
    {
        "name": "total_cost",
        "input": "open_set: Iterable[int], assignment: list[int]",
        "output": "float",
        "purpose": (
            "Total cost of a (open_set, single-source assignment) pair: sum "
            "of fixed costs over open warehouses plus sum of serve_cost(i,j) "
            "for each customer. Does NOT verify feasibility -- pair with "
            "tools['is_feasible'](to_solution(open_set, assignment)) for that."
        ),
    },
    # ----- (3) Construction / improvement -----
    {
        "name": "greedy_open_by_density",
        "input": "(no args)",
        "output": "(list[int], list[int])",
        "purpose": (
            "Construction heuristic: open warehouses in increasing order of "
            "fixed_cost / capacity until total open capacity >= total demand, "
            "then assign customers via greedy_serve_nearest. Returns "
            "(open_set, assignment) -- open_set is a sorted list of 0-indexed "
            "ids, assignment is a length-n list of 0-indexed warehouse ids "
            "(-1 where unassigned). Good warm start; pair with "
            "apply_swap_open_close or ilp_cwl to refine."
        ),
    },
    {
        "name": "greedy_serve_nearest",
        "input": "open_set: Iterable[int]",
        "output": "list[int]",
        "purpose": (
            "Given a set of open warehouses, assign each customer to the "
            "cheapest open warehouse with enough remaining capacity. "
            "Customers are processed largest-demand-first (hardest first). "
            "Unplaceable customers get -1. Returns a length-n list of "
            "0-indexed warehouse ids."
        ),
    },
    {
        "name": "apply_reassign_customer",
        "input": "assignment: list[int], customer: int (0-indexed), new_warehouse: int (0-indexed)",
        "output": "list[int] | None",
        "purpose": (
            "Return a NEW assignment with `customer` moved to `new_warehouse`. "
            "Returns None if the move would exceed new_warehouse's capacity "
            "given the rest of the assignment. Pure function -- the input is "
            "not mutated. NOTE: does not verify that new_warehouse is open; "
            "ensure it is in your open_set."
        ),
    },
    {
        "name": "apply_swap_open_close",
        "input": "open_set: Iterable[int], time_limit_s: float = 2.0",
        "output": "(list[int], list[int])",
        "purpose": (
            "Local search over warehouse OPEN/CLOSE status: try opening a "
            "closed warehouse, closing an open one, or swapping one for "
            "another; after each candidate move re-route customers with "
            "greedy_serve_nearest and accept if total cost improves. "
            "First-improvement, until time_limit_s elapses or no move helps. "
            "Returns (open_set, assignment)."
        ),
    },
    {
        "name": "to_solution",
        "input": "open_set: Iterable[int], assignment: list[int]",
        "output": "dict",
        "purpose": (
            "Convert a (open_set, single-source assignment) pair into the "
            "CO-Bench solution dict with keys 'total_cost', 'warehouse_open', "
            "'assignments'. Each customer's full demand is placed on its "
            "assigned warehouse (assignments[j][i] = demand[j] iff "
            "assignment[j] == i). Use this before returning from `solve`."
        ),
    },
    # ----- (4) Exact / heavy -----
    {
        "name": "ilp_cwl",
        "input": "time_limit_s: float = 10.0",
        "output": "dict | None",
        "purpose": (
            "Solve CWL (single-source variant) exactly via CBC (python-mip). "
            "Variables y[i] in {0,1} (open) and x[i,j] in {0,1} (j served by "
            "i), minimising sum F[i]*y[i] + sum C[i][j]*x[i,j] subject to "
            "'each customer served by exactly one warehouse' and 'warehouse "
            "load <= y[i] * capacity'. The CO-Bench cap*.txt benchmark admits "
            "single-source optima so binary x suffices. Returns a CO-Bench "
            "solution dict (ready to return from solve), or None if no "
            "feasible solution was found within the budget. Primary tool when "
            "the instance fits; use as the LNS subsolver otherwise."
        ),
    },
]
