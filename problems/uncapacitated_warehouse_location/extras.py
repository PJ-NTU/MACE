"""Per-problem extras for CO-Bench Uncapacitated Warehouse Location (UWL).

UWL is the simplest member of the facility-location family: pick a subset of
warehouses to open (each has a fixed opening cost) and assign each customer
ENTIRELY to one open warehouse. There is no capacity constraint, so once a
non-empty set S of warehouses is open, the optimal assignment is closed-form:
every customer picks its cheapest open warehouse. This makes the problem a
pure subset-selection problem over S = {open warehouses}.

Tool groups:
  (1) Queries:                warehouse_fixed_cost, serve_cost,
                              n_warehouses, n_customers, cheapest_warehouse_for
  (2) Feasibility primitives: cost_given_open, is_full_cover, solution_from_open
  (3) Construction / LS:      greedy_add_one, greedy_drop_one,
                              apply_swap_open_close
  (4) Heavy:                  ilp_uwl, lp_lower_bound

All are exposed under tools[...] and described in EXTRA_TOOLS_DESCRIPTION for
the LLM-facing prompt.
"""
from __future__ import annotations
import time
from typing import Iterable, Optional

from mip import Model, BINARY, CONTINUOUS, MINIMIZE, xsum, OptimizationStatus


def extra_tools(instance: dict) -> dict:
    """Factory: returns UWL-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench UWL load_data):
      - m:          number of potential warehouses (int)
      - n:          number of customers (int)
      - warehouses: list of dicts with 'fixed_cost' (and 'capacity' ignored)
      - customers:  list of dicts with 'costs' (list of m floats)
    """
    m = int(instance["m"])
    n = int(instance["n"])
    fixed = [float(instance["warehouses"][i]["fixed_cost"]) for i in range(m)]
    # c[j][i] = cost of serving customer j from warehouse i.
    c = [[float(instance["customers"][j]["costs"][i]) for i in range(m)]
         for j in range(n)]

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def warehouse_fixed_cost(i: int) -> float:
        if not (0 <= int(i) < m):
            raise ValueError(f"warehouse index {i} out of range [0, {m})")
        return fixed[int(i)]

    def serve_cost(i: int, j: int) -> float:
        if not (0 <= int(i) < m):
            raise ValueError(f"warehouse index {i} out of range [0, {m})")
        if not (0 <= int(j) < n):
            raise ValueError(f"customer index {j} out of range [0, {n})")
        return c[int(j)][int(i)]

    def n_warehouses() -> int:
        return m

    def n_customers() -> int:
        return n

    def cheapest_warehouse_for(j: int,
                               open_set: Optional[Iterable[int]] = None) -> int:
        """Index i of the warehouse with smallest c[j][i], restricted to
        `open_set` if given (otherwise all m warehouses). Returns None iff
        `open_set` is given and empty."""
        if not (0 <= int(j) < n):
            raise ValueError(f"customer index {j} out of range [0, {n})")
        if open_set is None:
            pool = range(m)
        else:
            pool = sorted({int(i) for i in open_set if 0 <= int(i) < m})
            if not pool:
                return None
        row = c[int(j)]
        best = None
        best_v = float("inf")
        for i in pool:
            if row[i] < best_v:
                best_v = row[i]
                best = i
        return best

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _normalize_open(open_set: Iterable[int]) -> list[int]:
        s = sorted({int(i) for i in open_set if 0 <= int(i) < m})
        return s

    def cost_given_open(open_set: Iterable[int]) -> float:
        """Total cost (fixed + assignment) when exactly `open_set` warehouses
        are open AND every customer is greedily assigned to its cheapest open
        warehouse. Returns +inf if open_set is empty (UWL is infeasible)."""
        opens = _normalize_open(open_set)
        if not opens:
            return float("inf")
        total = 0.0
        for i in opens:
            total += fixed[i]
        for j in range(n):
            row = c[j]
            best = row[opens[0]]
            for i in opens[1:]:
                v = row[i]
                if v < best:
                    best = v
            total += best
        return total

    def is_full_cover(open_set: Iterable[int]) -> bool:
        """True iff opening `open_set` lets every customer be served. In UWL
        (no capacity) this reduces to 'open_set is non-empty'."""
        return len(_normalize_open(open_set)) > 0

    def solution_from_open(open_set: Iterable[int]) -> dict:
        """Build a CO-Bench solution dict by assigning each customer to its
        cheapest open warehouse. Returns a dict with keys 'total_cost',
        'warehouse_open' (list[int] length m), 'assignments' (n x m list of
        lists). Returns None if open_set is empty (no feasible solution)."""
        opens = _normalize_open(open_set)
        if not opens:
            return None
        wh_open = [0] * m
        for i in opens:
            wh_open[i] = 1
        assignments = [[0] * m for _ in range(n)]
        total = 0.0
        for i in opens:
            total += fixed[i]
        for j in range(n):
            row = c[j]
            best_i = opens[0]
            best_v = row[best_i]
            for i in opens[1:]:
                v = row[i]
                if v < best_v:
                    best_v = v
                    best_i = i
            assignments[j][best_i] = 1
            total += best_v
        return {
            "total_cost": total,
            "warehouse_open": wh_open,
            "assignments": assignments,
        }

    # ==================================================================
    # (3) Construction / local search
    # ==================================================================
    def greedy_add_one(time_limit_s: float = 5.0) -> list[int]:
        """ADD heuristic: start with empty open set; repeatedly add the
        warehouse that yields the largest cost reduction (vs. opening one
        more warehouse than currently open). Stops when no addition improves
        cost or `time_limit_s` elapses. Returns the open set as a sorted
        list[int]. The first added warehouse is the one minimizing
        f_i + sum_j c[j][i] (i.e. opening i alone serves everyone)."""
        t0 = time.time()
        safety = 0.05
        # Step 0: pick the single warehouse minimizing f_i + sum_j c[j][i].
        best_i = 0
        best_cost = float("inf")
        for i in range(m):
            tot = fixed[i] + sum(c[j][i] for j in range(n))
            if tot < best_cost:
                best_cost = tot
                best_i = i
        opens = [best_i]
        cur_cost = best_cost
        # Maintain per-customer cheapest cost over current open set.
        best_cj = [c[j][best_i] for j in range(n)]

        not_open = set(range(m)) - {best_i}
        while not_open and (time.time() - t0) < time_limit_s - safety:
            best_delta = 0.0
            best_add = None
            for i in not_open:
                # adding i: pay f_i, but customer j's cost becomes min(best_cj[j], c[j][i])
                delta = fixed[i]
                for j in range(n):
                    v = c[j][i]
                    if v < best_cj[j]:
                        delta += v - best_cj[j]
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_add = i
            if best_add is None:
                break
            # commit the addition.
            opens.append(best_add)
            cur_cost += best_delta
            for j in range(n):
                v = c[j][best_add]
                if v < best_cj[j]:
                    best_cj[j] = v
            not_open.discard(best_add)
        return sorted(opens)

    def greedy_drop_one(time_limit_s: float = 5.0) -> list[int]:
        """DROP heuristic: start with all warehouses open; repeatedly remove
        the warehouse whose removal decreases cost the most. Stops when no
        removal improves cost or `time_limit_s` elapses (or only one
        warehouse remains, since UWL needs >=1 open). Returns the open set
        as a sorted list[int]."""
        t0 = time.time()
        safety = 0.05
        opens = set(range(m))
        cur_cost = cost_given_open(opens)
        while len(opens) > 1 and (time.time() - t0) < time_limit_s - safety:
            best_delta = 0.0
            best_drop = None
            opens_sorted = sorted(opens)
            for i in opens_sorted:
                trial = opens - {i}
                if not trial:
                    continue
                tc = cost_given_open(trial)
                delta = tc - cur_cost
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_drop = i
            if best_drop is None:
                break
            opens.discard(best_drop)
            cur_cost += best_delta
        return sorted(opens)

    def apply_swap_open_close(open_set: Iterable[int],
                              time_limit_s: float = 5.0) -> list[int]:
        """Pairwise swap local search: at each step try every (i_out, i_in)
        with i_out currently open and i_in currently closed, swap them, and
        commit the swap that reduces cost the most. Repeats until no
        improving swap exists or `time_limit_s` elapses. Returns the open
        set as a sorted list[int]. (Pure swap -- does NOT change the cardinality
        of the open set; combine with greedy_add_one / greedy_drop_one for
        cardinality moves.)"""
        t0 = time.time()
        safety = 0.05
        opens = set(_normalize_open(open_set))
        if not opens:
            return []
        cur_cost = cost_given_open(opens)
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            best_delta = 0.0
            best_swap = None
            opens_list = sorted(opens)
            closed_list = [i for i in range(m) if i not in opens]
            for i_out in opens_list:
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                for i_in in closed_list:
                    trial = (opens - {i_out}) | {i_in}
                    tc = cost_given_open(trial)
                    delta = tc - cur_cost
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_swap = (i_out, i_in)
            if best_swap is not None:
                i_out, i_in = best_swap
                opens.discard(i_out)
                opens.add(i_in)
                cur_cost += best_delta
                improved = True
        return sorted(opens)

    # ==================================================================
    # (4) Heavy
    # ==================================================================
    def _build_model(var_type, time_limit_s: float,
                     must_open: Optional[Iterable[int]] = None,
                     must_close: Optional[Iterable[int]] = None):
        mo = set(int(i) for i in must_open) if must_open else set()
        mc = set(int(i) for i in must_close) if must_close else set()
        mdl = Model(sense=MINIMIZE)
        mdl.verbose = 0
        mdl.max_seconds = float(time_limit_s)
        y = [mdl.add_var(var_type=var_type, lb=0.0, ub=1.0, name=f"y[{i}]")
             for i in range(m)]
        x = [[mdl.add_var(var_type=var_type, lb=0.0, ub=1.0,
                          name=f"x[{i},{j}]")
              for j in range(n)] for i in range(m)]
        mdl.objective = (
            xsum(fixed[i] * y[i] for i in range(m))
            + xsum(c[j][i] * x[i][j] for i in range(m) for j in range(n))
        )
        # each customer fully assigned to one warehouse
        for j in range(n):
            mdl += xsum(x[i][j] for i in range(m)) == 1, f"assign[{j}]"
        # cannot serve from closed warehouse: x[i,j] <= y[i] (the strong cut)
        for i in range(m):
            for j in range(n):
                mdl += x[i][j] <= y[i], f"link[{i},{j}]"
        for i in mo:
            if 0 <= i < m:
                mdl += y[i] == 1, f"open[{i}]"
        for i in mc:
            if 0 <= i < m:
                mdl += y[i] == 0, f"close[{i}]"
        return mdl, y, x

    def ilp_uwl(time_limit_s: float = 10.0,
                must_open: Optional[Iterable[int]] = None,
                must_close: Optional[Iterable[int]] = None) -> dict:
        """Solve the UWL ILP exactly (CBC). Variables: y[i] in {0,1} (open i),
        x[i,j] in {0,1} (i serves j). Minimizes sum f_i y_i + sum c[j,i] x[i,j]
        subject to (a) each customer has sum_i x[i,j] = 1, and (b) the strong
        link x[i,j] <= y[i]. Returns a CO-Bench solution dict ({'total_cost',
        'warehouse_open', 'assignments'}) or None if no solution was found.
        `must_open` / `must_close` let you fix subsets of y for LNS-style
        refinement."""
        mdl, y, x = _build_model(BINARY, time_limit_s, must_open, must_close)
        status = mdl.optimize()
        if status not in (OptimizationStatus.OPTIMAL,
                          OptimizationStatus.FEASIBLE):
            return None
        if mdl.num_solutions < 1:
            return None
        wh_open = [1 if (y[i].x or 0.0) > 0.5 else 0 for i in range(m)]
        assignments = [[0] * m for _ in range(n)]
        for j in range(n):
            # pick the warehouse with largest x[i,j] >= 0.5 (binary, so unique)
            chosen = None
            for i in range(m):
                if (x[i][j].x or 0.0) > 0.5:
                    chosen = i
                    break
            if chosen is None:
                # Numerical fallback: cheapest open warehouse.
                opens = [i for i in range(m) if wh_open[i] == 1]
                if not opens:
                    return None
                chosen = min(opens, key=lambda i: c[j][i])
            assignments[j][chosen] = 1
        # Use exact arithmetic for total_cost (more reliable than mdl.objective_value)
        total = 0.0
        for i in range(m):
            if wh_open[i] == 1:
                total += fixed[i]
        for j in range(n):
            for i in range(m):
                if assignments[j][i] == 1:
                    total += c[j][i]
        return {
            "total_cost": total,
            "warehouse_open": wh_open,
            "assignments": assignments,
        }

    def lp_lower_bound(time_limit_s: float = 10.0) -> float:
        """LP relaxation lower bound on the optimal UWL cost. Relaxes y[i] and
        x[i,j] to [0,1]. Tight on many instances; the integrality gap is often
        small. Returns +inf if the LP failed to solve."""
        mdl, _y, _x = _build_model(CONTINUOUS, time_limit_s)
        status = mdl.optimize()
        if status in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            try:
                return float(mdl.objective_value)
            except Exception:
                return float("inf")
        return float("inf")

    return {
        # (1) queries
        "warehouse_fixed_cost": warehouse_fixed_cost,
        "serve_cost": serve_cost,
        "n_warehouses": n_warehouses,
        "n_customers": n_customers,
        "cheapest_warehouse_for": cheapest_warehouse_for,
        # (2) feasibility primitives
        "cost_given_open": cost_given_open,
        "is_full_cover": is_full_cover,
        "solution_from_open": solution_from_open,
        # (3) construction / local search
        "greedy_add_one": greedy_add_one,
        "greedy_drop_one": greedy_drop_one,
        "apply_swap_open_close": apply_swap_open_close,
        # (4) heavy
        "ilp_uwl": ilp_uwl,
        "lp_lower_bound": lp_lower_bound,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "warehouse_fixed_cost",
        "input": "i: int",
        "output": "float",
        "purpose": "Fixed opening cost f_i of warehouse i. O(1).",
    },
    {
        "name": "serve_cost",
        "input": "i: int, j: int",
        "output": "float",
        "purpose": "Cost c[j][i] of serving customer j entirely from warehouse i. O(1).",
    },
    {
        "name": "n_warehouses",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of potential warehouses m.",
    },
    {
        "name": "n_customers",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of customers n.",
    },
    {
        "name": "cheapest_warehouse_for",
        "input": "j: int, open_set: Iterable[int] = None",
        "output": "int | None",
        "purpose": (
            "Index i of the warehouse with smallest c[j][i], restricted to "
            "`open_set` if given (otherwise over all m). Returns None iff "
            "`open_set` is given and empty. Useful for repair / reassignment."
        ),
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "cost_given_open",
        "input": "open_set: Iterable[int]",
        "output": "float",
        "purpose": (
            "Total cost (sum of fixed costs + greedy nearest-open-warehouse "
            "assignment costs) for opening exactly `open_set`. CLOSED-FORM in "
            "O(n * |open_set|): since UWL has no capacity, each customer "
            "trivially picks its cheapest open warehouse. Returns +inf if "
            "`open_set` is empty (UWL is infeasible)."
        ),
    },
    {
        "name": "is_full_cover",
        "input": "open_set: Iterable[int]",
        "output": "bool",
        "purpose": (
            "True iff opening `open_set` lets every customer be served. In UWL "
            "(no capacity) this is equivalent to 'open_set is non-empty'."
        ),
    },
    {
        "name": "solution_from_open",
        "input": "open_set: Iterable[int]",
        "output": "dict | None",
        "purpose": (
            "Build the full CO-Bench solution dict ('total_cost', "
            "'warehouse_open', 'assignments') from an open set by assigning "
            "each customer to its cheapest open warehouse. Returns None if "
            "open_set is empty. Use this to convert local-search states into "
            "the dict that solve() must return."
        ),
    },
    # ----- (3) Construction / local search -----
    {
        "name": "greedy_add_one",
        "input": "time_limit_s: float = 5.0",
        "output": "list[int]",
        "purpose": (
            "ADD heuristic: start with a single best warehouse, then "
            "repeatedly add the warehouse that yields the largest cost "
            "reduction, until no addition improves cost. Returns the open set "
            "as a sorted list[int]. O(m^2 * n) worst case; very fast in "
            "practice."
        ),
    },
    {
        "name": "greedy_drop_one",
        "input": "time_limit_s: float = 5.0",
        "output": "list[int]",
        "purpose": (
            "DROP heuristic: start with ALL warehouses open, then repeatedly "
            "remove the warehouse whose removal decreases cost the most. "
            "Stops when no removal improves cost (or only one warehouse "
            "remains). Returns the open set as a sorted list[int]. "
            "Complementary to greedy_add_one -- often finds a different local "
            "optimum."
        ),
    },
    {
        "name": "apply_swap_open_close",
        "input": "open_set: Iterable[int], time_limit_s: float = 5.0",
        "output": "list[int]",
        "purpose": (
            "Best-improvement swap local search: at each step, considers all "
            "(i_out, i_in) pairs with i_out currently open and i_in currently "
            "closed, and commits the swap that reduces cost the most. Repeats "
            "until no improving swap exists or `time_limit_s` elapses. "
            "Preserves cardinality of the open set, so combine with "
            "greedy_add_one / greedy_drop_one to also adjust cardinality."
        ),
    },
    # ----- (4) Heavy -----
    {
        "name": "ilp_uwl",
        "input": "time_limit_s: float = 10.0, must_open: Iterable[int] = None, must_close: Iterable[int] = None",
        "output": "dict | None",
        "purpose": (
            "Solve UWL exactly via the strong ILP formulation (CBC). "
            "Variables: y[i] in {0,1} (open i), x[i,j] in {0,1} (i serves j). "
            "Constraints: sum_i x[i,j] = 1 for each customer j, and "
            "x[i,j] <= y[i] (the *strong* link cut, much tighter than "
            "sum_j x[i,j] <= n*y[i]). `must_open`/`must_close` fix subsets of "
            "y for LNS-style refinement of an incumbent. Returns a CO-Bench "
            "solution dict or None if no solution found within time_limit_s."
        ),
    },
    {
        "name": "lp_lower_bound",
        "input": "time_limit_s: float = 10.0",
        "output": "float",
        "purpose": (
            "LP relaxation lower bound (relaxes y[i], x[i,j] to [0,1]). Often "
            "very tight for UWL because the strong link x[i,j] <= y[i] gives "
            "an LP gap that is small in practice. Useful as an optimality "
            "gauge: if heuristic_cost / lp_lower_bound is close to 1, you can "
            "stop early."
        ),
    },
]
