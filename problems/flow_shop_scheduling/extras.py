"""Per-problem extras for CO-Bench Permutation Flow Shop Scheduling.

Provides primitive building blocks so the LLM can compose construction +
local-search heuristics without re-deriving the flow-shop recurrence,
NEH insertion, Johnson's rule, etc.

Instance schema (from CO-Bench load_data):
  - n (int):                number of jobs
  - m (int):                number of machines
  - matrix (n x m list):    matrix[job_0_indexed][machine_0_indexed] = proc time
  - upper_bound, lower_bound (int): instance bounds (informational)

CONVENTION:
  All permutations exposed by these tools are **1-indexed** lists -- exactly
  the shape eval_func expects for `solution['job_sequence']`. So an LLM can
  feed the output of `neh_construct()` directly into the solution dict.

Tool groups:
  Tier 1 (queries):       processing_time, n_jobs, n_machines
  Tier 2 (feasibility):   simulate_makespan, completion_times
  Tier 3 (construct/LS):  neh_construct, johnson_2machine_construct,
                          apply_swap_search, apply_insertion_search,
                          swap_delta_makespan
  Tier 4 (heavy / exact): ilp_flow_shop (disjunctive big-M; small n only)
"""
from __future__ import annotations
import random
import time
from typing import Optional

import numpy as np


def extra_tools(instance: dict) -> dict:
    """Factory: returns flow-shop-specific tool callables given the loaded instance.

    All returned tools close over a precomputed numpy processing-time matrix P
    of shape (n, m), where P[j, k] is the processing time of job j (0-indexed)
    on machine k (0-indexed). The 1-indexed/0-indexed conversion is handled
    internally; the LLM only ever sees 1-indexed permutations.
    """
    n: int = int(instance["n"])
    m: int = int(instance["m"])
    matrix = instance["matrix"]
    # P[j, k] = processing time of job j (0-indexed) on machine k (0-indexed)
    P = np.asarray(matrix, dtype=np.int64)
    if P.shape != (n, m):
        raise ValueError(f"matrix shape {P.shape} does not match (n={n}, m={m})")

    # Precompute total job processing time (sum across machines). Used by NEH
    # to sort jobs in decreasing total-processing-time order before insertion.
    job_totals = P.sum(axis=1)  # shape (n,)

    # -------- internal helpers (0-indexed) --------
    def _validate_perm_1idx(perm) -> list[int]:
        """Convert/validate a 1-indexed permutation; return 0-indexed list."""
        if not isinstance(perm, (list, tuple)):
            raise TypeError(f"permutation must be list/tuple, got {type(perm).__name__}")
        if len(perm) != n:
            raise ValueError(f"permutation length {len(perm)} != n={n}")
        seen = set()
        out: list[int] = []
        for v in perm:
            iv = int(v)
            if not (1 <= iv <= n):
                raise ValueError(f"job id {iv} out of range [1, {n}]")
            if iv in seen:
                raise ValueError(f"job id {iv} appears twice")
            seen.add(iv)
            out.append(iv - 1)
        return out

    def _makespan_0idx(perm0: list[int]) -> int:
        """Compute makespan via the classical flow-shop DP. O(len(perm) * m)."""
        L = len(perm0)
        if L == 0:
            return 0
        # C[k] = completion time on machine k for the current job in the sweep
        C = np.zeros(m, dtype=np.int64)
        for j_pos in range(L):
            j = perm0[j_pos]
            C[0] += P[j, 0]
            for k in range(1, m):
                a = C[k - 1]
                b = C[k]
                C[k] = (a if a > b else b) + P[j, k]
        return int(C[-1])

    def _completion_table_0idx(perm0: list[int]) -> np.ndarray:
        """Full completion-time table of shape (L, m). Standard recurrence."""
        L = len(perm0)
        C = np.zeros((L, m), dtype=np.int64)
        if L == 0:
            return C
        for i in range(L):
            j = perm0[i]
            for k in range(m):
                if i == 0 and k == 0:
                    C[i, k] = P[j, 0]
                elif i == 0:
                    C[i, k] = C[i, k - 1] + P[j, k]
                elif k == 0:
                    C[i, k] = C[i - 1, k] + P[j, 0]
                else:
                    a = C[i - 1, k]
                    b = C[i, k - 1]
                    C[i, k] = (a if a > b else b) + P[j, k]
        return C

    # ==================================================================
    # Tier 1: queries
    # ==================================================================
    def n_jobs() -> int:
        return n

    def n_machines() -> int:
        return m

    def processing_time(job: int, machine: int) -> int:
        """1-indexed job, 0-indexed machine (machines are usually 0..m-1)."""
        if not (1 <= int(job) <= n):
            raise ValueError(f"job={job} out of [1, {n}]")
        if not (0 <= int(machine) < m):
            raise ValueError(f"machine={machine} out of [0, {m})")
        return int(P[int(job) - 1, int(machine)])

    # ==================================================================
    # Tier 2: feasibility / objective primitives
    # ==================================================================
    def simulate_makespan(permutation: list) -> int:
        """Makespan (C_max) of a 1-indexed full permutation. O(n*m)."""
        perm0 = _validate_perm_1idx(permutation)
        return _makespan_0idx(perm0)

    def completion_times(permutation: list, machine: Optional[int] = None):
        """Completion-time table for a 1-indexed permutation.
        If machine is None: returns the full (n, m) list-of-lists.
        Else: returns a list of length n giving completion on that machine
        for each position in the permutation.
        """
        perm0 = _validate_perm_1idx(permutation)
        C = _completion_table_0idx(perm0)
        if machine is None:
            return C.tolist()
        if not (0 <= int(machine) < m):
            raise ValueError(f"machine={machine} out of [0, {m})")
        return C[:, int(machine)].tolist()

    # ==================================================================
    # Tier 3: construction
    # ==================================================================
    def _best_insertion_position(partial0: list[int], job0: int) -> tuple[list[int], int]:
        """Insert job0 in every position of partial0; return (best_perm, best_cmax).
        Each evaluation is O((L+1)*m); total O((L+1)^2 * m). Standard NEH step."""
        L = len(partial0)
        best_perm = None
        best_cmax = None
        for pos in range(L + 1):
            cand = partial0[:pos] + [job0] + partial0[pos:]
            cmx = _makespan_0idx(cand)
            if best_cmax is None or cmx < best_cmax:
                best_cmax = cmx
                best_perm = cand
        return best_perm, int(best_cmax)

    def neh_construct() -> list[int]:
        """NEH heuristic (Nawaz-Enscore-Ham 1983) for permutation flow shop.
        1. Sort jobs by decreasing total processing time (sum over machines).
        2. Insert jobs one by one in the position that minimizes partial makespan.
        Returns a 1-indexed full permutation. Complexity O(n^3 * m). Classical
        and very strong starting heuristic."""
        order0 = list(np.argsort(-job_totals).tolist())  # decreasing total time
        if n == 0:
            return []
        partial = [order0[0]]
        for j in order0[1:]:
            partial, _ = _best_insertion_position(partial, j)
        return [v + 1 for v in partial]

    def johnson_2machine_construct() -> Optional[list[int]]:
        """Johnson's rule -- OPTIMAL for the 2-machine permutation flow shop
        (and a reasonable heuristic when applied to extreme machines for m>2,
        but here we keep it strict). Returns a 1-indexed permutation, or None
        if m != 2.
        Rule: split jobs into set A (where P[j,0] <= P[j,1]) and set B (rest).
              A sorted by P[j,0] ascending; B sorted by P[j,1] descending;
              concatenate A then B. O(n log n)."""
        if m != 2:
            return None
        A: list[tuple[int, int]] = []  # (key, job0)
        B: list[tuple[int, int]] = []
        for j in range(n):
            p1 = int(P[j, 0])
            p2 = int(P[j, 1])
            if p1 <= p2:
                A.append((p1, j))
            else:
                B.append((-p2, j))  # negate so ascending sort = descending p2
        A.sort()
        B.sort()
        order0 = [j for _, j in A] + [j for _, j in B]
        return [v + 1 for v in order0]

    # ==================================================================
    # Tier 3: local search & deltas
    # ==================================================================
    def swap_delta_makespan(permutation: list, i: int, j: int) -> int:
        """Change in makespan if positions i and j (0-indexed positions within
        the permutation list) are swapped. Negative => improvement.
        Computed by full recomputation: O(n*m). Unlike TSP 2-opt, flow-shop
        makespan does not admit an O(1) endpoint delta in general."""
        perm0 = _validate_perm_1idx(permutation)
        L = len(perm0)
        if not (0 <= i < L and 0 <= j < L):
            raise ValueError(f"positions out of range: i={i}, j={j}, n={L}")
        if i == j:
            return 0
        cur = _makespan_0idx(perm0)
        new = perm0.copy()
        new[i], new[j] = new[j], new[i]
        return int(_makespan_0idx(new)) - int(cur)

    def apply_swap_search(perm: list, time_limit_s: float = 5.0,
                          first_improvement: bool = True) -> list:
        """Pairwise-swap local search. For each pair of positions (i, j), swap
        and keep if makespan improves. Returns improved 1-indexed permutation.
        Each pass O(n^2 * n * m) = O(n^3 * m) in the worst case (since each
        evaluation re-simulates). Use modest time_limit_s for large n."""
        perm0 = _validate_perm_1idx(perm)
        L = len(perm0)
        if L < 2:
            return [v + 1 for v in perm0]
        cur = _makespan_0idx(perm0)
        t0 = time.time()
        safety = 0.05
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            best_delta = 0
            best_ij = None
            for i in range(L - 1):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                for j in range(i + 1, L):
                    cand = perm0.copy()
                    cand[i], cand[j] = cand[j], cand[i]
                    new_cmax = _makespan_0idx(cand)
                    delta = new_cmax - cur
                    if delta < 0:
                        if first_improvement:
                            perm0 = cand
                            cur = new_cmax
                            improved = True
                            break
                        elif delta < best_delta:
                            best_delta = delta
                            best_ij = (i, j)
                if first_improvement and improved:
                    break
            if (not first_improvement) and best_ij is not None:
                i, j = best_ij
                perm0[i], perm0[j] = perm0[j], perm0[i]
                cur = cur + best_delta
                improved = True
        return [v + 1 for v in perm0]

    def apply_insertion_search(perm: list, time_limit_s: float = 5.0,
                               first_improvement: bool = True) -> list:
        """Insertion (or-opt with segment length 1) local search: try removing
        each job and reinserting it at every other position, keep the best
        improving move. Often stronger than swap for flow shop because it
        matches NEH's neighborhood. Each pass O(n^2 * m) using full re-sim per
        evaluation, so this is O(n^3 * m) per outer iteration in the worst case."""
        perm0 = _validate_perm_1idx(perm)
        L = len(perm0)
        if L < 2:
            return [v + 1 for v in perm0]
        cur = _makespan_0idx(perm0)
        t0 = time.time()
        safety = 0.05
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            best_delta = 0
            best_move = None
            for i in range(L):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                job = perm0[i]
                rest = perm0[:i] + perm0[i + 1:]
                for k in range(L):  # k = new position in the full perm
                    if k == i:
                        continue
                    cand = rest[:k] + [job] + rest[k:]
                    new_cmax = _makespan_0idx(cand)
                    delta = new_cmax - cur
                    if delta < 0:
                        if first_improvement:
                            perm0 = cand
                            cur = new_cmax
                            improved = True
                            break
                        elif delta < best_delta:
                            best_delta = delta
                            best_move = (i, k, job)
                if first_improvement and improved:
                    break
            if (not first_improvement) and best_move is not None:
                i, k, job = best_move
                rest = perm0[:i] + perm0[i + 1:]
                perm0 = rest[:k] + [job] + rest[k:]
                cur = cur + best_delta
                improved = True
        return [v + 1 for v in perm0]

    # ==================================================================
    # Tier 4: ILP (disjunctive big-M; for small instances only)
    # ==================================================================
    def ilp_flow_shop(time_limit_s: float = 30.0) -> Optional[list[int]]:
        """Solve the permutation flow shop exactly (or to best-found within
        time limit) via a disjunctive big-M MILP using python-mip + CBC.

        Decision variables:
          x[i, j] in {0,1} : 1 iff job i precedes job j somewhere in the seq
                             (here we use position-assignment form instead --
                              see below for the actual formulation we use).
          We use the position-assignment formulation:
            z[i, r] in {0,1} : 1 iff job i is at position r (r in 0..n-1)
            C[r, k] >= 0     : completion of position r on machine k

        Constraints:
          - Each job assigned to exactly one position; each position has one job.
          - C[0, 0] = sum_i z[i, 0] * P[i, 0]
          - C[r, 0] = C[r-1, 0] + sum_i z[i, r] * P[i, 0]
          - C[0, k] = C[0, k-1] + sum_i z[i, 0] * P[i, k]
          - C[r, k] >= C[r-1, k] + sum_i z[i, r] * P[i, k]
          - C[r, k] >= C[r, k-1] + sum_i z[i, r] * P[i, k]
          - minimize C[n-1, m-1]

        Returns a 1-indexed permutation, or None if the solver finds nothing.

        WARNING: scales poorly. Practical only for n up to ~15-20. For larger
        instances use neh_construct + apply_insertion_search.
        """
        try:
            from mip import Model, BINARY, MINIMIZE, CONTINUOUS, xsum, OptimizationStatus
        except ImportError:
            return None

        mdl = Model(sense=MINIMIZE)
        mdl.verbose = 0
        mdl.max_seconds = float(time_limit_s)

        z = {(i, r): mdl.add_var(var_type=BINARY, name=f"z_{i}_{r}")
             for i in range(n) for r in range(n)}
        C = {(r, k): mdl.add_var(var_type=CONTINUOUS, lb=0.0, name=f"C_{r}_{k}")
             for r in range(n) for k in range(m)}

        # assignment constraints
        for i in range(n):
            mdl += xsum(z[i, r] for r in range(n)) == 1, f"job_{i}_once"
        for r in range(n):
            mdl += xsum(z[i, r] for i in range(n)) == 1, f"pos_{r}_once"

        # processing-time-at-position expressions
        def p_at(r: int, k: int):
            return xsum(int(P[i, k]) * z[i, r] for i in range(n))

        # initial cells
        mdl += C[0, 0] >= p_at(0, 0), "c00"
        for k in range(1, m):
            mdl += C[0, k] >= C[0, k - 1] + p_at(0, k), f"c0_{k}"
        for r in range(1, n):
            mdl += C[r, 0] >= C[r - 1, 0] + p_at(r, 0), f"c{r}_0"
        # general recurrence
        for r in range(1, n):
            for k in range(1, m):
                mdl += C[r, k] >= C[r - 1, k] + p_at(r, k), f"prev_pos_{r}_{k}"
                mdl += C[r, k] >= C[r, k - 1] + p_at(r, k), f"prev_mch_{r}_{k}"

        mdl.objective = C[n - 1, m - 1]
        status = mdl.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if mdl.num_solutions < 1:
            return None

        # extract permutation
        perm0: list[int] = [-1] * n
        for r in range(n):
            for i in range(n):
                v = z[i, r].x
                if v is not None and v > 0.5:
                    perm0[r] = i
                    break
        if any(v < 0 for v in perm0):
            return None
        return [v + 1 for v in perm0]

    # ==================================================================
    # Solution-dict builder + one-shot solver
    # ==================================================================
    def make_solution(permutation) -> dict:
        """Wrap a 1-indexed permutation into the EXACT dict shape eval_func
        expects: {'job_sequence': list[int]}. Validates length n and the
        permutation property; raises ValueError otherwise. Use on the
        output of neh_construct() / apply_insertion_search() / ilp_flow_shop()
        so you never return the wrong dict shape."""
        perm0 = _validate_perm_1idx(permutation)
        return {"job_sequence": [v + 1 for v in perm0]}

    def solve_default(time_limit_s: float = 5.0) -> dict:
        """ONE-SHOT STRONG SOLVER. Returns the complete solution dict
        {'job_sequence': list[int]} ready to return directly.

        Strategy: build an NEH warm start (Nawaz-Enscore-Ham 1983, the
        gold-standard heuristic for permutation flow shop, typically
        within a few percent of optimum), then polish with
        apply_insertion_search under the remaining time budget. Returns
        the best permutation found wrapped in the solution dict.

        Use as the FIRST thing your solve() function calls. ONE LINE:
            return tools['solve_default'](time_limit_s=5)
        """
        if n == 0:
            return {"job_sequence": []}
        perm = neh_construct()
        if time_limit_s > 0.5:
            perm = apply_insertion_search(
                perm, time_limit_s=max(0.5, time_limit_s - 0.1),
                first_improvement=True,
            )
        return make_solution(perm)

    return {
        # one-shot + builder (CALL FIRST)
        "solve_default": solve_default,
        "make_solution": make_solution,
        # heavy / exact
        "ilp_flow_shop": ilp_flow_shop,
        # construction
        "neh_construct": neh_construct,
        "johnson_2machine_construct": johnson_2machine_construct,
        # local search
        "apply_insertion_search": apply_insertion_search,
        "apply_swap_search": apply_swap_search,
        "swap_delta_makespan": swap_delta_makespan,
        # feasibility / queries
        "simulate_makespan": simulate_makespan,
        "completion_times": completion_times,
        "processing_time": processing_time,
        "n_jobs": n_jobs,
        "n_machines": n_machines,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- ONE-SHOT STRONG SOLVER (call this first!) -----
    {
        "name": "solve_default",
        "input": "time_limit_s: float = 5.0",
        "output": "dict {'job_sequence': list[int]} (1-indexed permutation)",
        "purpose": (
            "RECOMMENDED START: returns a complete solution dict ready to return "
            "directly. Builds an NEH warm start (gold-standard flow-shop "
            "heuristic, typically within a few percent of optimum), then polishes "
            "with apply_insertion_search under the remaining budget. ONE LINE: "
            "`return tools['solve_default'](time_limit_s=5)`."
        ),
    },
    {
        "name": "make_solution",
        "input": "permutation: list[int] (1-indexed, length n)",
        "output": "dict {'job_sequence': list[int]}",
        "purpose": (
            "Build the EXACT solution dict shape eval_func wants from a 1-indexed "
            "permutation. Validates length n and the permutation property. Use on "
            "the output of neh_construct() / apply_insertion_search() / "
            "ilp_flow_shop() so you never return the wrong dict shape."
        ),
    },
    # ----- Tier 1: queries -----
    {
        "name": "n_jobs",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of jobs (n). Convenience accessor for instance['n'].",
    },
    {
        "name": "n_machines",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of machines (m). Convenience accessor for instance['m'].",
    },
    {
        "name": "processing_time",
        "input": "job: int (1-indexed), machine: int (0-indexed, in [0, m))",
        "output": "int",
        "purpose": (
            "Processing time of `job` on `machine`. Job is 1-indexed to match "
            "the solution `job_sequence` convention; machine is 0-indexed."
        ),
    },
    # ----- Tier 2: feasibility / objective primitives -----
    {
        "name": "simulate_makespan",
        "input": "permutation: list[int] (1-indexed, length n)",
        "output": "int",
        "purpose": (
            "Makespan (completion time of the last job on the last machine) "
            "via the classical flow-shop DP. O(n*m). Use this in tight inner "
            "loops -- faster and side-effect-free compared to tools['objective']."
        ),
    },
    {
        "name": "completion_times",
        "input": "permutation: list[int] (1-indexed), machine: int | None = None",
        "output": "list[list[int]]  (n x m)  OR  list[int]  (length n) if machine given",
        "purpose": (
            "Full completion-time table for a 1-indexed permutation, or just "
            "one column if `machine` is specified. Useful for analyzing "
            "bottleneck machines and idle gaps. O(n*m)."
        ),
    },
    # ----- Tier 3: construction -----
    {
        "name": "neh_construct",
        "input": "(no args)",
        "output": "list[int]  (1-indexed permutation of length n)",
        "purpose": (
            "NEH heuristic (Nawaz-Enscore-Ham 1983) -- the standard high-quality "
            "starting solution for permutation flow shop. Sort jobs by total "
            "processing time descending, then insert one by one at the position "
            "that minimizes the partial makespan. Returns a full 1-indexed "
            "permutation. O(n^3 * m). Often within a few percent of optimum; "
            "always run this first then improve with apply_insertion_search."
        ),
    },
    {
        "name": "johnson_2machine_construct",
        "input": "(no args)",
        "output": "list[int] | None  (1-indexed permutation, or None if m != 2)",
        "purpose": (
            "Johnson's rule -- OPTIMAL for the 2-machine permutation flow shop. "
            "Returns None when m != 2 (the rule does not generalize cleanly). "
            "O(n log n). Use only when n_machines() == 2."
        ),
    },
    # ----- Tier 3: local search -----
    {
        "name": "swap_delta_makespan",
        "input": "permutation: list[int] (1-indexed), i: int, j: int  (0-indexed positions)",
        "output": "int  (new_makespan - old_makespan; negative = improvement)",
        "purpose": (
            "Change in makespan from swapping the jobs at positions i and j. "
            "Computed by full O(n*m) re-simulation -- unlike TSP 2-opt, flow "
            "shop has no O(1) endpoint delta. Convenient when you want to "
            "evaluate a candidate move without overwriting the current permutation."
        ),
    },
    {
        "name": "apply_swap_search",
        "input": "perm: list[int] (1-indexed), time_limit_s: float = 5.0, first_improvement: bool = True",
        "output": "list[int]  (1-indexed)",
        "purpose": (
            "Pairwise-swap local search: try every (i, j) pair, accept "
            "improving swaps until no improvement or time runs out. "
            "O(n^2) moves per pass, each move costs O(n*m) re-sim. Weaker "
            "than insertion search but a useful diversifier."
        ),
    },
    {
        "name": "apply_insertion_search",
        "input": "perm: list[int] (1-indexed), time_limit_s: float = 5.0, first_improvement: bool = True",
        "output": "list[int]  (1-indexed)",
        "purpose": (
            "Insertion (or-opt-1) local search: pull each job out and try every "
            "other insertion position; accept improving moves. Matches NEH's "
            "neighborhood and is typically the strongest single local search "
            "for permutation flow shop. Each outer pass is O(n^3 * m) in the "
            "worst case. Use this as the workhorse after neh_construct."
        ),
    },
    # ----- Tier 4: heavy / exact -----
    {
        "name": "ilp_flow_shop",
        "input": "time_limit_s: float = 30.0",
        "output": "list[int] | None  (1-indexed optimal/best-found permutation)",
        "purpose": (
            "Exact MILP for permutation flow shop using python-mip + CBC, in "
            "position-assignment form (z[i, r] = job i at position r, with "
            "completion-time recurrence constraints). Returns the optimal "
            "permutation (or best-found within time_limit_s), else None. "
            "WARNING: scales poorly -- practical only for n up to ~15-20. "
            "For larger n, use neh_construct + apply_insertion_search."
        ),
    },
]
