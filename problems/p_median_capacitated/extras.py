"""Per-problem extras for Capacitated P-Median (CPM).

Provides primitive building blocks so the LLM can compose construction /
repair / local-search heuristics for CPM without re-deriving distance
bookkeeping, capacity accounting, or basic ILP plumbing.

Tool groups:
  (1) Queries:        facility_capacity, customer_demand, cost,
                      p, n_facilities, n_customers
  (2) Feasibility:    facility_load, is_within_capacity,
                      unassigned_customers
  (3) Construction /
      improvement:    greedy_p_picks_by_distance,
                      assignment_by_nearest_feasible,
                      apply_swap_open_close,
                      apply_reassign_customer,
                      to_solution
  (4) Exact / heavy:  ilp_cpm

Index conventions (IMPORTANT):
  - The problem stores customers with 1-based `customer_id` and the
    CO-Bench `eval_func` expects `medians` and `assignments` to be those
    1-based IDs. Internally these tools use 0-based positional indices
    (`i` for a facility candidate = customer position, `j` for a customer
    position) because positional arithmetic is the natural unit for
    distance / demand arrays. Use `to_solution(open_set, assignment)` to
    convert the 0-indexed (open_set, assignment) pair into the 1-based
    CO-Bench solution dict before returning from `solve`.

Representation:
  - `open_set` (a.k.a. the chosen medians) is a Python iterable of
    0-indexed facility positions, of length exactly `p`. (Each customer
    is also a candidate facility, so positions live in [0, n).)
  - `assignment` is a list[int] of length n, with `assignment[j] = i`
    meaning customer j is served by the facility at position i. Entries
    outside [0, n) are treated as unassigned.
"""
from __future__ import annotations
import math
import time
from typing import Iterable, List, Optional, Tuple

from mip import Model, BINARY, MINIMIZE, xsum, OptimizationStatus


def extra_tools(instance: dict) -> dict:
    """Factory: returns CPM-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench p-median - capacitated load_data, one case):
      - best_known: float
      - n:          int, number of customers (also = number of candidate facilities)
      - p:          int, number of medians to open
      - Q:          float, capacity of each median (uniform)
      - customers:  list[tuple] of length n: (customer_id, x, y, demand)
    """
    n: int = int(instance["n"])
    p_val: int = int(instance["p"])
    Q: float = float(instance["Q"])
    customers = instance["customers"]

    if len(customers) != n:
        raise ValueError(f"customers length {len(customers)} != n={n}")

    # Position-indexed (0-based) arrays
    cust_ids: List[int] = [int(c[0]) for c in customers]
    xs: List[float] = [float(c[1]) for c in customers]
    ys: List[float] = [float(c[2]) for c in customers]
    D: List[float] = [float(c[3]) for c in customers]  # demand[j]

    # Precompute floored Euclidean cost matrix (matches eval_func semantics):
    # cost[i][j] = floor( sqrt((x_i - x_j)^2 + (y_i - y_j)^2) ).
    cost_mat: List[List[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(n):
            dx = xi - xs[j]
            dy = yi - ys[j]
            cost_mat[i][j] = int(math.floor(math.sqrt(dx * dx + dy * dy)))

    def _i(i: int) -> int:
        i = int(i)
        if not (0 <= i < n):
            raise ValueError(f"facility position {i} out of range [0, {n})")
        return i

    def _j(j: int) -> int:
        j = int(j)
        if not (0 <= j < n):
            raise ValueError(f"customer position {j} out of range [0, {n})")
        return j

    def _validate_assignment(assignment: Iterable[int]) -> List[int]:
        a = list(assignment)
        if len(a) != n:
            raise ValueError(f"assignment has length {len(a)}, expected {n}")
        return a

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def facility_capacity(i: int) -> float:
        """Capacity of facility i (uniform Q for all candidates in CPM)."""
        _i(i)
        return Q

    def customer_demand(j: int) -> float:
        """Demand of customer at position j (0-indexed)."""
        return D[_j(j)]

    def cost(i: int, j: int) -> int:
        """Assignment cost of customer j being served by facility i:
        floor(Euclidean distance), matching the eval_func semantics."""
        return cost_mat[_i(i)][_j(j)]

    def p() -> int:
        """Number of medians that must be opened (problem parameter p)."""
        return p_val

    def n_facilities() -> int:
        """Number of candidate facility positions (== n, since every
        customer is a candidate median)."""
        return n

    def n_customers() -> int:
        """Number of customers n."""
        return n

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def facility_load(i: int, assignment: Iterable[int]) -> float:
        """Total demand assigned to facility i (0-indexed) under
        `assignment`. Entries outside [0, n) are ignored."""
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

    def is_within_capacity(open_set: Iterable[int],
                           assignment: Iterable[int]) -> bool:
        """True iff every facility in `open_set` has total assigned demand
        <= Q (within a 1e-6 tolerance, matching eval_func), AND every
        customer in `assignment` is mapped to a facility in open_set.
        Returns False if any customer is unassigned (entry outside [0, n))
        or assigned to a facility not in open_set."""
        a = _validate_assignment(assignment)
        opens = {int(i) for i in open_set if 0 <= int(i) < n}
        load = {i: 0.0 for i in opens}
        for j, aj in enumerate(a):
            try:
                v = int(aj)
            except (TypeError, ValueError):
                return False
            if v not in opens:
                return False
            load[v] += D[j]
        for i in opens:
            if load[i] > Q + 1e-6:
                return False
        return True

    def unassigned_customers(assignment: Iterable[int]) -> List[int]:
        """0-indexed customer positions whose assignment entry is missing
        or out of range (None, negative, or >= n). Useful during
        incremental construction."""
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
            if v < 0 or v >= n:
                out.append(j)
        return out

    # ==================================================================
    # (3) Construction / improvement
    # ==================================================================
    def greedy_p_picks_by_distance() -> List[int]:
        """Pick p facility positions greedily to minimise summed
        nearest-facility cost across customers (a furthest-first / k-medoids
        style seeding):
          1) Open the position with smallest sum_j cost(i, j).
          2) Repeatedly open the position that most reduces
             sum_j min(cost(picked, j), cost(candidate, j)).
        Returns a sorted list of p 0-indexed positions. Does NOT consider
        capacity; pair with assignment_by_nearest_feasible to route demand
        (which may leave some customers unassigned if the chosen p sites
        cannot fit all demand)."""
        if p_val <= 0 or p_val > n:
            raise ValueError(f"p={p_val} not in [1, {n}]")
        # Seed: the position minimising the total cost to all customers.
        sums = [sum(cost_mat[i]) for i in range(n)]
        first = min(range(n), key=lambda i: sums[i])
        picks: List[int] = [first]
        best_to = list(cost_mat[first])  # best_to[j] = min cost so far to any pick
        while len(picks) < p_val:
            best_cand = -1
            best_total = float("inf")
            for cand in range(n):
                if cand in picks:
                    continue
                total = 0.0
                row = cost_mat[cand]
                for j in range(n):
                    v = row[j] if row[j] < best_to[j] else best_to[j]
                    total += v
                if total < best_total:
                    best_total = total
                    best_cand = cand
            if best_cand < 0:
                break
            picks.append(best_cand)
            row = cost_mat[best_cand]
            for j in range(n):
                if row[j] < best_to[j]:
                    best_to[j] = row[j]
        return sorted(picks)

    def assignment_by_nearest_feasible(open_set: Iterable[int]) -> List[int]:
        """Assign each customer to its cheapest open facility that still
        has enough remaining capacity. Customers are processed in
        decreasing-demand order (hardest first). If no open facility can
        fit a customer, the slot is set to -1. Returns a length-n list of
        0-indexed facility positions."""
        opens = sorted({int(i) for i in open_set if 0 <= int(i) < n})
        if not opens:
            return [-1] * n
        load = {i: 0.0 for i in opens}
        assignment = [-1] * n
        order = sorted(range(n), key=lambda j: -D[j])
        for j in order:
            best_i = -1
            best_c = float("inf")
            for i in opens:
                if load[i] + D[j] <= Q + 1e-9:
                    c = cost_mat[i][j]
                    if c < best_c:
                        best_c = c
                        best_i = i
            if best_i >= 0:
                assignment[j] = best_i
                load[best_i] += D[j]
            # else leave as -1 (unassigned)
        return assignment

    def apply_reassign_customer(assignment: Iterable[int],
                                customer: int,
                                new_facility: int) -> Optional[List[int]]:
        """Return a NEW assignment with customer `customer` (0-indexed)
        moved to facility `new_facility` (0-indexed). Returns None if the
        move would exceed new_facility's capacity given the rest of the
        assignment. The input list is never mutated. Does NOT check that
        new_facility is open -- the caller is responsible for that."""
        a = _validate_assignment(assignment)
        j = _j(customer)
        new_i = _i(new_facility)
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
        if new_load > Q + 1e-9:
            return None
        out = list(a)
        out[j] = new_i
        return out

    def _routing_cost(open_set: List[int], assignment: List[int]) -> float:
        total = 0.0
        for j, aj in enumerate(assignment):
            try:
                i0 = int(aj)
            except (TypeError, ValueError):
                return float("inf")
            if 0 <= i0 < n:
                total += cost_mat[i0][j]
            else:
                return float("inf")
        return total

    def apply_swap_open_close(open_set: Iterable[int],
                              t_limit: float = 2.0) -> Tuple[List[int], List[int]]:
        """Local search over WHICH p facilities to open. At each step, try
        swapping one currently-open facility with one currently-closed
        candidate; re-route customers via assignment_by_nearest_feasible
        and accept the first improving swap. The number of open facilities
        is always held at p. First-improvement, until t_limit seconds
        elapse or no swap improves cost. Returns (open_set, assignment)
        where open_set is a sorted list of p 0-indexed positions and
        assignment is the corresponding length-n list."""
        t0 = time.time()
        safety = 0.05
        opens = sorted({int(i) for i in open_set if 0 <= int(i) < n})
        if len(opens) != p_val:
            raise ValueError(f"open_set has {len(opens)} elements, expected p={p_val}")

        def _score(opens_list):
            assn = assignment_by_nearest_feasible(opens_list)
            if any(a < 0 for a in assn):
                return float("inf"), assn
            return _routing_cost(opens_list, assn), assn

        best_cost, best_assn = _score(opens)
        improved = True
        while improved and (time.time() - t0) < t_limit - safety:
            improved = False
            closed = [i for i in range(n) if i not in opens]
            for k_close in list(opens):
                if (time.time() - t0) >= t_limit - safety:
                    break
                for k_open in closed:
                    if (time.time() - t0) >= t_limit - safety:
                        break
                    cand = sorted([i for i in opens if i != k_close] + [k_open])
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
        """Convert a (0-indexed open_set, 0-indexed assignment) pair into
        the CO-Bench solution dict expected by eval_func:
          - 'objective':   total floored-distance routing cost
          - 'medians':     list of p customer_ids (1-based) of the
                           chosen facility positions
          - 'assignments': list[int] of length n, the customer_id (1-based)
                           that each customer (in customer order) is
                           assigned to.
        Use this before returning from `solve`."""
        a = _validate_assignment(assignment)
        opens = sorted({int(i) for i in open_set if 0 <= int(i) < n})
        medians_ids = [cust_ids[i] for i in opens]
        assignments_ids: List[int] = []
        total = 0
        for j, aj in enumerate(a):
            try:
                i0 = int(aj)
            except (TypeError, ValueError):
                i0 = -1
            if 0 <= i0 < n:
                assignments_ids.append(cust_ids[i0])
                total += cost_mat[i0][j]
            else:
                # Sentinel: 0 is never a valid customer_id (they are 1..n),
                # so the eval_func will (correctly) reject this solution.
                assignments_ids.append(0)
        return {
            "objective": int(total),
            "medians": medians_ids,
            "assignments": assignments_ids,
        }

    # ==================================================================
    # (4) Exact / heavy: ILP
    # ==================================================================
    def ilp_cpm(time_limit_s: float = 10.0) -> Optional[dict]:
        """Solve CPM exactly via CBC (python-mip).

        Variables:
          - y[i] in {0,1}  : facility position i is opened (a median)
          - x[i,j] in {0,1}: customer j is served by facility i
        Objective: minimize sum_{i,j} cost[i][j] * x[i,j]
        Constraints:
          - sum_i y[i] == p                       (exactly p medians)
          - sum_i x[i,j] == 1   for each customer j (each customer served)
          - x[i,j] <= y[i]                        (only open facilities serve)
          - sum_j D[j]*x[i,j] <= Q * y[i]         (capacity)

        Returns a CO-Bench solution dict ready to return from `solve`, or
        None if no feasible solution was found within the budget."""
        model = Model(sense=MINIMIZE)
        model.verbose = 0
        model.max_seconds = float(time_limit_s)
        y = [model.add_var(var_type=BINARY, name=f"y_{i}") for i in range(n)]
        x = [[model.add_var(var_type=BINARY, name=f"x_{i}_{j}")
              for j in range(n)] for i in range(n)]
        model.objective = xsum(cost_mat[i][j] * x[i][j]
                               for i in range(n) for j in range(n))
        model += xsum(y[i] for i in range(n)) == p_val, "p_medians"
        for j in range(n):
            model += xsum(x[i][j] for i in range(n)) == 1, f"cust_{j}"
        for i in range(n):
            for j in range(n):
                model += x[i][j] <= y[i], f"link_{i}_{j}"
            model += xsum(D[j] * x[i][j] for j in range(n)) <= Q * y[i], f"cap_{i}"
        status = model.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if model.num_solutions < 1:
            return None
        opens: List[int] = []
        for i in range(n):
            val = y[i].x
            if val is not None and val > 0.5:
                opens.append(i)
        if len(opens) != p_val:
            return None
        assignment = [-1] * n
        for j in range(n):
            for i in opens:
                val = x[i][j].x
                if val is not None and val > 0.5:
                    assignment[j] = i
                    break
            if assignment[j] < 0:
                return None
        return to_solution(opens, assignment)

    return {
        # (1) queries
        "facility_capacity": facility_capacity,
        "customer_demand": customer_demand,
        "cost": cost,
        "p": p,
        "n_facilities": n_facilities,
        "n_customers": n_customers,
        # (2) feasibility primitives
        "facility_load": facility_load,
        "is_within_capacity": is_within_capacity,
        "unassigned_customers": unassigned_customers,
        # (3) construction / improvement
        "greedy_p_picks_by_distance": greedy_p_picks_by_distance,
        "assignment_by_nearest_feasible": assignment_by_nearest_feasible,
        "apply_swap_open_close": apply_swap_open_close,
        "apply_reassign_customer": apply_reassign_customer,
        "to_solution": to_solution,
        # (4) exact
        "ilp_cpm": ilp_cpm,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "facility_capacity",
        "input": "i: int (0-indexed facility position)",
        "output": "float",
        "purpose": (
            "Capacity of facility i. In CPM all candidates share the same "
            "uniform capacity Q, so this returns Q regardless of i (and "
            "validates that i is in [0, n))."
        ),
    },
    {
        "name": "customer_demand",
        "input": "j: int (0-indexed customer position)",
        "output": "float",
        "purpose": (
            "Demand of customer at position j (the 4th column of "
            "customers[j])."
        ),
    },
    {
        "name": "cost",
        "input": "i: int (0-indexed facility), j: int (0-indexed customer)",
        "output": "int",
        "purpose": (
            "Assignment cost of customer j served by facility i: floor of "
            "the Euclidean distance, matching eval_func's "
            "math.floor(sqrt((cx-mx)^2 + (cy-my)^2)) semantics. Precomputed "
            "n x n table; O(1) lookup."
        ),
    },
    {
        "name": "p",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of medians that must be opened (problem parameter p).",
    },
    {
        "name": "n_facilities",
        "input": "(no args)",
        "output": "int",
        "purpose": (
            "Number of candidate facility positions. In CPM every customer "
            "is also a candidate median, so this equals n_customers()."
        ),
    },
    {
        "name": "n_customers",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of customers n.",
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "facility_load",
        "input": "i: int (0-indexed), assignment: list[int]",
        "output": "float",
        "purpose": (
            "Total demand assigned to facility i under `assignment`. "
            "`assignment` is a length-n list of 0-indexed facility positions "
            "(entries outside [0, n) are ignored)."
        ),
    },
    {
        "name": "is_within_capacity",
        "input": "open_set: Iterable[int], assignment: list[int]",
        "output": "bool",
        "purpose": (
            "True iff (a) every customer in `assignment` is mapped to a "
            "facility in `open_set`, and (b) every facility's total assigned "
            "demand is <= Q (within 1e-6). Quick neighbour filter before "
            "calling tools['is_feasible'] (which delegates to eval_func)."
        ),
    },
    {
        "name": "unassigned_customers",
        "input": "assignment: list[int]",
        "output": "list[int]",
        "purpose": (
            "0-indexed customer positions whose assignment entry is missing "
            "(None) or out of range. Use during incremental construction to "
            "detect customers still to be placed."
        ),
    },
    # ----- (3) Construction / improvement -----
    {
        "name": "greedy_p_picks_by_distance",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Pick exactly p facility positions greedily to minimise the "
            "summed nearest-facility cost across customers (k-medoids style "
            "seeding). Capacity is NOT considered here; pair with "
            "assignment_by_nearest_feasible to route demand. If the chosen "
            "p sites cannot fit all demand, assignment_by_nearest_feasible "
            "will leave some customers as -1 and you should refine via "
            "apply_swap_open_close or ilp_cpm."
        ),
    },
    {
        "name": "assignment_by_nearest_feasible",
        "input": "open_set: Iterable[int]",
        "output": "list[int]",
        "purpose": (
            "Greedy router: assign each customer (largest-demand first) to "
            "its cheapest open facility with enough remaining capacity. "
            "Unplaceable customers get -1. Returns a length-n list of "
            "0-indexed facility positions."
        ),
    },
    {
        "name": "apply_swap_open_close",
        "input": "open_set: Iterable[int], t_limit: float = 2.0",
        "output": "(list[int], list[int])",
        "purpose": (
            "Local search holding |open_set| fixed at p: try swapping each "
            "currently-open facility with each closed one; re-route via "
            "assignment_by_nearest_feasible and accept the first improving "
            "swap. Repeats until t_limit elapses or no swap improves. "
            "Returns (open_set, assignment)."
        ),
    },
    {
        "name": "apply_reassign_customer",
        "input": "assignment: list[int], customer: int (0-indexed), new_facility: int (0-indexed)",
        "output": "list[int] | None",
        "purpose": (
            "Return a NEW assignment with `customer` moved to `new_facility`. "
            "Returns None if the move would exceed new_facility's capacity. "
            "Pure function -- the input is not mutated. Does not verify that "
            "new_facility is open; ensure it is in your open_set."
        ),
    },
    {
        "name": "to_solution",
        "input": "open_set: Iterable[int], assignment: list[int]",
        "output": "dict",
        "purpose": (
            "Convert a (0-indexed open_set, 0-indexed assignment) pair into "
            "the CO-Bench solution dict with keys 'objective', 'medians', "
            "'assignments'. Translates 0-indexed positions back to the "
            "problem's 1-based customer_ids that eval_func expects. Use this "
            "before returning from `solve`."
        ),
    },
    # ----- (4) Exact / heavy -----
    {
        "name": "ilp_cpm",
        "input": "time_limit_s: float = 10.0",
        "output": "dict | None",
        "purpose": (
            "Solve CPM exactly via CBC (python-mip). Variables y[i] in {0,1} "
            "(facility i open) and x[i,j] in {0,1} (customer j served by i), "
            "minimising sum cost[i][j]*x[i,j] subject to: exactly p medians "
            "open, each customer served exactly once, x[i,j] <= y[i], and "
            "sum_j demand[j]*x[i,j] <= Q*y[i]. Returns a CO-Bench solution "
            "dict (ready to return from solve) or None if no feasible "
            "solution was found within the budget. Primary tool when the "
            "instance is small enough; otherwise use as an LNS subsolver."
        ),
    },
]
