"""Per-problem extras for CO-Bench Uncapacitated p-Median (UPM).

Problem:
  Given an n-vertex graph with all-pairs shortest distance matrix D and a
  number p, pick exactly p vertices ('medians' / open facilities). Each of
  the n customers is implicitly assigned to its NEAREST open median. The
  objective is the sum over all customers of the distance to its nearest
  open median. No capacities, no fixed costs.

Vertex sets:
  In this benchmark, the facility set and customer set are the SAME -- both
  are the n vertices of the graph. So n_facilities == n_customers == n.
  Vertices are 1-indexed in the solution dict (medians), but the internal
  distance matrix is 0-indexed (instance['dist'][i][j]).

Tool groups (the LLM may use any subset, or roll its own):
  (1) Queries:                cost, p, n_facilities, n_customers
  (2) Feasibility primitives: cost_given_open, validate_open_count
  (3) Construction / LS:      greedy_add_one_until_p, apply_swap_one_for_one,
                              apply_interchange_LK
  (4) Heavy:                  ilp_upm, lp_lower_bound

The classical local search for UPM is the SWAP-1-for-1 (Teitz-Bart /
'interchange') heuristic -- which is what `apply_swap_one_for_one` and
`apply_interchange_LK` implement. UPM's LP relaxation is famously tight,
so `lp_lower_bound` is a useful sanity check.
"""
from __future__ import annotations
import time
from typing import Iterable, Optional

import numpy as np


def extra_tools(instance: dict) -> dict:
    """Factory: returns UPM-specific tool callables for one instance.

    Instance schema (from CO-Bench p-median uncapacitated load_data):
      - n:    int, number of vertices.
      - m:    int, number of edges (informational).
      - p:    int, number of medians to choose.
      - dist: list[list[float]], n x n all-pairs shortest path matrix.
    """
    n: int = int(instance["n"])
    p_val: int = int(instance["p"])
    D = np.asarray(instance["dist"], dtype=np.float64)  # (n, n), 0-indexed

    # In this problem facilities == customers == V. Expose both names so
    # the LLM doesn't have to guess.
    n_fac = n
    n_cus = n

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def cost(i: int, j: int) -> float:
        """Shortest-path distance from vertex i (1-indexed) to vertex j (1-indexed)."""
        if not (1 <= int(i) <= n) or not (1 <= int(j) <= n):
            raise ValueError(f"vertices must be in [1, {n}], got i={i}, j={j}")
        return float(D[int(i) - 1, int(j) - 1])

    def p() -> int:
        """Required number of open medians."""
        return p_val

    def n_facilities() -> int:
        """Number of candidate facility sites (== n)."""
        return n_fac

    def n_customers() -> int:
        """Number of customers to serve (== n)."""
        return n_cus

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _open_to_zero_indexed_array(open_set: Iterable[int]) -> np.ndarray:
        """Normalize a user-supplied open-facility set (1-indexed) to a
        unique, in-range int array of 0-indexed positions. Raises on
        out-of-range or non-int entries."""
        try:
            arr = np.asarray(list(open_set), dtype=np.int64)
        except Exception as e:
            raise ValueError(f"open_set must be iterable of ints, got {open_set!r}") from e
        if arr.size == 0:
            return arr
        if arr.min() < 1 or arr.max() > n:
            raise ValueError(f"open_set entries must be in [1, {n}], got range "
                             f"[{int(arr.min())}, {int(arr.max())}]")
        return arr - 1

    def cost_given_open(open_set: Iterable[int]) -> float:
        """Closed-form UPM cost when `open_set` (1-indexed vertex ids) are the
        chosen medians: each customer goes to its NEAREST open vertex.

        Returns +inf if open_set is empty (no median => unreachable) or if
        any customer has all-inf distances to the open set.
        Does NOT enforce |open_set| == p -- use validate_open_count for that.
        """
        open0 = _open_to_zero_indexed_array(open_set)
        if open0.size == 0:
            return float("inf")
        # nearest-open distance per customer = min over open columns
        sub = D[:, open0]                # (n, |open|)
        nearest = sub.min(axis=1)        # (n,)
        if not np.all(np.isfinite(nearest)):
            return float("inf")
        return float(nearest.sum())

    def validate_open_count(open_set: Iterable[int]) -> tuple[bool, Optional[str]]:
        """Check that `open_set` has exactly p distinct vertices in [1, n].
        Returns (True, None) if OK, else (False, reason)."""
        try:
            lst = list(open_set)
        except Exception as e:
            return False, f"open_set not iterable: {e}"
        if len(lst) != p_val:
            return False, f"expected exactly p={p_val} medians, got {len(lst)}"
        if any(not isinstance(v, int) for v in lst):
            return False, "all medians must be int"
        if any(v < 1 or v > n for v in lst):
            return False, f"each median must be in [1, {n}]"
        if len(set(lst)) != p_val:
            return False, "medians must be distinct"
        return True, None

    # ==================================================================
    # (3) Construction / local search
    # ==================================================================
    def greedy_add_one_until_p() -> list[int]:
        """Greedy construction: start with open = empty, repeatedly add the
        single not-yet-open vertex that most reduces the current assignment
        cost, until |open| = p. O(p * n^2) time. Returns a 1-indexed list of
        exactly p distinct vertices.

        This is the standard greedy seeding heuristic for p-median; it is
        usually within a few percent of optimal on the Beasley pmed
        instances and makes a strong warm start for swap-based local search.
        """
        # 'nearest[c]' = distance from customer c (0-indexed) to its nearest
        # currently-open facility. Initially +inf (no facility open).
        nearest = np.full(n, np.inf, dtype=np.float64)
        open_mask = np.zeros(n, dtype=bool)
        chosen: list[int] = []
        for _ in range(p_val):
            # For each candidate j (not yet open), the new nearest would be
            # min(nearest, D[:, j]). We pick j minimizing sum of that.
            best_j = -1
            best_total = float("inf")
            # Vectorized: compute candidate totals one column-block at a time
            # (cheap since n is at most a few thousand for Beasley).
            cand = np.where(~open_mask)[0]
            # For each cand j, candidate_nearest = minimum(nearest, D[:, j]).
            # sum_j = sum over customers of candidate_nearest.
            # Do it as a vectorized (n, |cand|) min.
            cand_cols = D[:, cand]                                # (n, |cand|)
            new_nearest = np.minimum(nearest[:, None], cand_cols)  # broadcasting
            totals = new_nearest.sum(axis=0)                       # (|cand|,)
            k = int(np.argmin(totals))
            best_j = int(cand[k])
            best_total = float(totals[k])
            chosen.append(best_j + 1)  # 1-indexed
            open_mask[best_j] = True
            nearest = np.minimum(nearest, D[:, best_j])
        return chosen

    def _swap_pass(open_arr: np.ndarray, time_limit_s: float,
                   first_improvement: bool) -> tuple[np.ndarray, float, bool]:
        """One pass of swap-1-for-1: try replacing each open vertex with each
        not-open vertex. Returns (new_open, new_cost, improved)."""
        n_open = open_arr.size
        # Boolean open mask, useful to enumerate the closed set.
        open_mask = np.zeros(n, dtype=bool)
        open_mask[open_arr] = True
        closed = np.where(~open_mask)[0]

        # nearest1[c] = distance to closest open;
        # nearest2[c] = distance to second-closest open;
        # arg1[c]    = index INTO open_arr (0..n_open-1) of closest open.
        sub = D[:, open_arr]                          # (n, n_open)
        # argsort once is cheaper than two min calls when n_open <= ~200.
        order = np.argsort(sub, axis=1)               # (n, n_open)
        idx_n = np.arange(n)
        arg1 = order[:, 0]
        nearest1 = sub[idx_n, arg1]
        if n_open >= 2:
            arg2 = order[:, 1]
            nearest2 = sub[idx_n, arg2]
        else:
            nearest2 = np.full(n, np.inf)
        cur_cost = float(nearest1.sum())

        best_delta = 0.0
        best_in = -1   # closed vertex to bring in (0-indexed)
        best_out_pos = -1  # position within open_arr to remove

        t0 = time.time()
        safety = 0.02

        for out_pos in range(n_open):
            if (time.time() - t0) >= time_limit_s - safety:
                break
            # If we remove open_arr[out_pos], customers whose arg1==out_pos
            # fall back to nearest2; others keep nearest1. Then we add a new
            # facility 'cand' and they reassign if D[c,cand] < their fallback.
            removed_mask = (arg1 == out_pos)
            # fallback distance per customer if we just removed out_pos
            fb = np.where(removed_mask, nearest2, nearest1)
            # If after removal someone has inf, only a candidate that
            # reaches them can save it -- new_dist below handles that.
            for cand in closed:
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                d_cand = D[:, cand]
                new_dist = np.minimum(fb, d_cand)
                if not np.all(np.isfinite(new_dist)):
                    continue
                new_cost = float(new_dist.sum())
                delta = new_cost - cur_cost
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_in = int(cand)
                    best_out_pos = out_pos
                    if first_improvement:
                        break
            if first_improvement and best_in >= 0:
                break

        if best_in < 0:
            return open_arr, cur_cost, False
        new_open = open_arr.copy()
        new_open[best_out_pos] = best_in
        return new_open, cur_cost + best_delta, True

    def apply_swap_one_for_one(open_set: Iterable[int],
                               time_limit_s: float = 5.0,
                               first_improvement: bool = True) -> list[int]:
        """Classical Teitz-Bart 'interchange' local search: repeatedly try
        replacing one open vertex with one currently-closed vertex, accepting
        any improvement. Returns a 1-indexed list of exactly p medians.

        This is the workhorse heuristic for p-median and usually gets within
        a couple of percent of optimum on the Beasley instances when seeded
        from greedy_add_one_until_p.
        """
        open0 = _open_to_zero_indexed_array(open_set)
        if open0.size != p_val:
            raise ValueError(f"open_set must have exactly p={p_val} entries, "
                             f"got {open0.size}")
        if len(set(open0.tolist())) != open0.size:
            raise ValueError("open_set must have distinct entries")
        open_arr = open0.copy()
        t0 = time.time()
        safety = 0.05
        while (time.time() - t0) < time_limit_s - safety:
            remaining = max(0.0, time_limit_s - (time.time() - t0) - safety)
            open_arr, _, improved = _swap_pass(open_arr, remaining, first_improvement)
            if not improved:
                break
        return sorted(int(v + 1) for v in open_arr.tolist())

    def apply_interchange_LK(open_set: Iterable[int],
                             time_limit_s: float = 5.0) -> list[int]:
        """Lin-Kernighan-style chained swap: each outer iteration does one
        best-improvement swap-1-for-1 (so chained moves can escape regions
        that first-improvement gets stuck in). Returns a 1-indexed list of p
        medians.

        Empirically slower per iteration than `apply_swap_one_for_one(...,
        first_improvement=True)` but often finds better solutions with the
        same time budget on harder instances.
        """
        return apply_swap_one_for_one(open_set, time_limit_s=time_limit_s,
                                      first_improvement=False)

    # ==================================================================
    # (4) Heavy: ILP and LP-bound
    # ==================================================================
    def _build_mip(integer: bool, time_limit_s: float):
        """Build the standard UPM IP/LP:
            min  sum_{i,j} D[i,j] * x[i,j]
            s.t. sum_j x[i,j] = 1               for each customer i
                 x[i,j] <= y[j]                  for each i,j
                 sum_j y[j] = p
                 x,y in {0,1} (or [0,1] for LP)
        """
        from mip import Model, BINARY, CONTINUOUS, MINIMIZE, xsum, OptimizationStatus
        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)
        vtype = BINARY if integer else CONTINUOUS
        # y[j] = 1 iff facility j (0-indexed) is open
        y = [m.add_var(var_type=vtype, lb=0.0, ub=1.0, name=f"y[{j}]") for j in range(n)]
        # x[i][j] = 1 iff customer i is assigned to facility j
        x = [[m.add_var(var_type=vtype, lb=0.0, ub=1.0, name=f"x[{i},{j}]")
              for j in range(n)] for i in range(n)]

        m.objective = xsum(float(D[i, j]) * x[i][j]
                           for i in range(n) for j in range(n)
                           if np.isfinite(D[i, j]))
        # Forbid assignment along inf edges (shouldn't happen post Floyd-Warshall, but safe).
        for i in range(n):
            for j in range(n):
                if not np.isfinite(D[i, j]):
                    m += x[i][j] == 0
        # Each customer fully assigned
        for i in range(n):
            m += xsum(x[i][j] for j in range(n)) == 1
        # x[i][j] <= y[j]
        for i in range(n):
            for j in range(n):
                m += x[i][j] <= y[j]
        # Exactly p facilities open
        m += xsum(y[j] for j in range(n)) == p_val
        return m, x, y, OptimizationStatus

    def ilp_upm(time_limit_s: float = 30.0) -> Optional[list[int]]:
        """Solve UPM exactly (or to the best feasible found within
        time_limit_s) via the standard MIP using python-mip / CBC.

        Returns a sorted 1-indexed list of p median vertices, or None if no
        feasible solution was found within the time budget.
        """
        m, _x, y, OptimizationStatus = _build_mip(integer=True,
                                                  time_limit_s=time_limit_s)
        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None
        chosen = [j + 1 for j in range(n)
                  if y[j].x is not None and y[j].x > 0.5]
        # If MIP gave a slightly off count, trim/pad would mask the bug -- return None instead.
        if len(chosen) != p_val:
            return None
        return sorted(chosen)

    def lp_lower_bound(time_limit_s: float = 30.0) -> Optional[float]:
        """Solve the LP relaxation of UPM (drop y, x in {0,1}). The optimal
        LP value is a valid LOWER BOUND on any integer solution; for p-median
        the LP gap is famously small (often < 1%), so this is a good
        certificate of how close your incumbent is to optimal.

        Returns the LP optimal value, or None if the LP solver fails.
        """
        from mip import OptimizationStatus
        m, _x, _y, _OS = _build_mip(integer=False, time_limit_s=time_limit_s)
        status = m.optimize()
        if status not in (_OS.OPTIMAL, _OS.FEASIBLE):
            return None
        try:
            return float(m.objective_value)
        except Exception:
            return None

    return {
        "cost": cost,
        "p": p,
        "n_facilities": n_facilities,
        "n_customers": n_customers,
        "cost_given_open": cost_given_open,
        "validate_open_count": validate_open_count,
        "greedy_add_one_until_p": greedy_add_one_until_p,
        "apply_swap_one_for_one": apply_swap_one_for_one,
        "apply_interchange_LK": apply_interchange_LK,
        "ilp_upm": ilp_upm,
        "lp_lower_bound": lp_lower_bound,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "cost",
        "input": "i: int, j: int  (vertices, 1-indexed)",
        "output": "float",
        "purpose": (
            "Shortest-path distance between vertex i and vertex j (1-indexed). "
            "O(1) lookup into the precomputed all-pairs matrix."
        ),
    },
    {
        "name": "p",
        "input": "(no args)",
        "output": "int",
        "purpose": "Required number of open medians for this instance.",
    },
    {
        "name": "n_facilities",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of candidate facility sites (equals n; same set as customers).",
    },
    {
        "name": "n_customers",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of customers (equals n; same set as facilities).",
    },
    # ----- Feasibility primitives -----
    {
        "name": "cost_given_open",
        "input": "open_set: Iterable[int]  (1-indexed vertex ids)",
        "output": "float",
        "purpose": (
            "Closed-form UPM objective when `open_set` are the open medians: "
            "each customer is auto-assigned to its NEAREST open vertex and "
            "the total distance is returned. O(n * |open|). Returns +inf if "
            "open_set is empty or some customer is unreachable. Does NOT "
            "enforce |open_set| == p -- pair with validate_open_count."
        ),
    },
    {
        "name": "validate_open_count",
        "input": "open_set: Iterable[int]",
        "output": "(bool, str | None)",
        "purpose": (
            "Check that `open_set` has exactly p distinct ints in [1, n]. "
            "Returns (True, None) if OK, else (False, reason)."
        ),
    },
    # ----- Construction / local search -----
    {
        "name": "greedy_add_one_until_p",
        "input": "(no args)",
        "output": "list[int]  (1-indexed, length p)",
        "purpose": (
            "Greedy construction: start with no medians open, repeatedly add "
            "the single closed vertex that most reduces the assignment cost "
            "until exactly p are open. O(p * n^2). Strong warm start for "
            "swap-based local search."
        ),
    },
    {
        "name": "apply_swap_one_for_one",
        "input": "open_set: Iterable[int], time_limit_s: float = 5.0, first_improvement: bool = True",
        "output": "list[int]  (1-indexed, length p)",
        "purpose": (
            "Classical Teitz-Bart 'interchange' local search: try replacing "
            "each open vertex with each closed vertex; accept any improvement. "
            "first_improvement=True is faster per pass; False (best-improvement) "
            "often converges to a better local optimum. Re-runs passes until "
            "no improvement or time runs out. Typically reaches within a few "
            "percent of optimum on Beasley pmed instances."
        ),
    },
    {
        "name": "apply_interchange_LK",
        "input": "open_set: Iterable[int], time_limit_s: float = 5.0",
        "output": "list[int]  (1-indexed, length p)",
        "purpose": (
            "Best-improvement chained swap-1-for-1 (Lin-Kernighan flavor). "
            "Slower per iteration than apply_swap_one_for_one with "
            "first_improvement=True, but tends to find better solutions on "
            "harder instances within the same time budget."
        ),
    },
    # ----- Heavy -----
    {
        "name": "ilp_upm",
        "input": "time_limit_s: float = 30.0",
        "output": "list[int] | None  (1-indexed, length p)",
        "purpose": (
            "Solve the full UPM MIP exactly with python-mip / CBC (or return "
            "the best feasible found within time_limit_s). Variables: y[j] in "
            "{0,1} for open-facility, x[i,j] in {0,1} for assignment. Returns "
            "None on infeasibility / no solution found. Use as primary solver "
            "on small / moderate n, or as a polish stage after local search."
        ),
    },
    {
        "name": "lp_lower_bound",
        "input": "time_limit_s: float = 30.0",
        "output": "float | None",
        "purpose": (
            "Solve the LP relaxation (x, y in [0,1]) of the UPM MIP. Returns "
            "a valid LOWER BOUND on the integer optimum. The UPM LP gap is "
            "famously small (often < 1%), so this is a cheap certificate of "
            "how close your incumbent is to optimal -- e.g. if your incumbent "
            "objective is within 1% of lp_lower_bound() you can stop."
        ),
    },
]
