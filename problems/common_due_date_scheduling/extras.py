"""Per-problem extras for CO-Bench Common Due Date Scheduling.

Provides building blocks so the LLM can compose construction + local-search
heuristics for the restricted single-machine common-due-date problem without
reinventing penalty evaluation, V-shape ordering, swap/insertion moves, etc.

Solution schema: {"schedule": list[int]} -- a 1-based permutation of jobs.

Theory:
  An optimal schedule has the V-shape property (Panwalkar-Smith, 1982):
  jobs completing strictly before the due date are sequenced in NON-INCREASING
  order of p_i / alpha_i (earliness ratio); jobs starting at or after the due
  date are sequenced in NON-DECREASING order of p_i / beta_i (tardiness ratio).
  Whether the first late job straddles d or starts at d depends on the
  instance; both options are evaluated in v_shape_construct.

Tool groups:
  (1) Queries:        processing_time, penalty_early, penalty_late,
                      due_date, n_jobs
  (2) Feasibility:    compute_total_penalty
  (3) Construction:   edd_construct, wspt_construct, v_shape_construct
  (4) Improvement:    apply_swap_2opt, apply_insertion_search
  (5) Heavy:          ilp_cdd
"""
from __future__ import annotations
import time
from typing import Optional, Sequence


def extra_tools(instance: dict) -> dict:
    """Factory: returns CDDS-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench Common due date scheduling load_data):
      - jobs: list[(p, a, b)]  processing time, earliness coef, tardiness coef
      - h:    float            due-date fraction (default 0.6)
    """
    jobs = [tuple(j) for j in instance["jobs"]]
    h = float(instance.get("h", 0.6))
    n = len(jobs)
    p_arr = [int(j[0]) for j in jobs]
    a_arr = [int(j[1]) for j in jobs]
    b_arr = [int(j[2]) for j in jobs]
    total_p = sum(p_arr)
    d_val = int(total_p * h)  # matches eval_func: floor for non-negative sum

    # ==================================================================
    # (1) Queries -- O(1)
    # ==================================================================
    def processing_time(j: int) -> int:
        """Processing time p_j of job j (1-based index)."""
        if not (1 <= j <= n):
            raise ValueError(f"job index j={j} out of range [1, {n}]")
        return int(p_arr[j - 1])

    def penalty_early(j: int) -> int:
        """Earliness penalty coefficient alpha_j of job j (1-based index)."""
        if not (1 <= j <= n):
            raise ValueError(f"job index j={j} out of range [1, {n}]")
        return int(a_arr[j - 1])

    def penalty_late(j: int) -> int:
        """Tardiness penalty coefficient beta_j of job j (1-based index)."""
        if not (1 <= j <= n):
            raise ValueError(f"job index j={j} out of range [1, {n}]")
        return int(b_arr[j - 1])

    def due_date() -> int:
        """Common due date d = floor(sum(p) * h)."""
        return int(d_val)

    def n_jobs() -> int:
        """Number of jobs in this instance."""
        return int(n)

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def compute_total_penalty(permutation: Sequence[int],
                              start_time: int = 0) -> int:
        """Total earliness+tardiness penalty for processing jobs in
        `permutation` order starting at `start_time`. `permutation` must be a
        permutation of [1, n] (1-based job indices). Does NOT mutate input.
        Matches CO-Bench eval_func exactly when start_time=0."""
        if len(permutation) != n:
            raise ValueError(
                f"permutation has length {len(permutation)}, expected {n}"
            )
        if sorted(permutation) != list(range(1, n + 1)):
            raise ValueError(
                f"permutation must be a permutation of 1..{n}"
            )
        c = int(start_time)
        total = 0
        for idx in permutation:
            p, a, b = jobs[idx - 1]
            c += p
            if c < d_val:
                total += a * (d_val - c)
            elif c > d_val:
                total += b * (c - d_val)
        return int(total)

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def edd_construct() -> list:
        """Earliest-Due-Date: with a common due date all jobs share the same
        deadline, so EDD degenerates -- as a stable proxy we sort by
        non-decreasing processing time (SPT), which on a single machine with
        common d minimizes total completion-time deviation in the absence of
        weights. Returns a 1-based permutation."""
        order = sorted(range(n), key=lambda i: (p_arr[i], i))
        return [i + 1 for i in order]

    def wspt_construct() -> list:
        """Weighted Shortest Processing Time: sort jobs by ascending
        p_i / max(beta_i, 1) so high-tardiness-cost / short jobs go first.
        Useful when most jobs end up tardy (small h). Returns a 1-based
        permutation."""
        def key(i):
            b = b_arr[i] if b_arr[i] > 0 else 1
            return (p_arr[i] / b, i)
        order = sorted(range(n), key=key)
        return [i + 1 for i in order]

    def v_shape_construct() -> list:
        """Panwalkar-Smith V-shape construction.

        Split jobs into an `early` set (completes at or before d) and a `late`
        set (starts at or after d). Within early, sort by NON-INCREASING
        p_i / alpha_i so the smallest-alpha-per-unit-p job ends just before d.
        Within late, sort by NON-DECREASING p_i / beta_i so the smallest-beta-
        per-unit-p job is pushed furthest right.

        Job assignment to early vs late uses a greedy fill: add jobs in
        decreasing p_i / alpha_i order while sum of processing times stays
        <= d; remainder goes to late. This is a known good heuristic but not
        provably optimal. Returns a 1-based permutation.
        """
        if n == 0:
            return []
        # Sort all jobs by decreasing p_i / alpha_i (alpha=0 => +inf, goes first).
        def early_key(i):
            a = a_arr[i]
            if a <= 0:
                return (float("inf"), -i)
            return (p_arr[i] / a, -i)
        order_by_alpha_desc = sorted(range(n), key=early_key, reverse=True)

        early: list[int] = []
        late: list[int] = []
        used = 0
        for i in order_by_alpha_desc:
            if used + p_arr[i] <= d_val:
                early.append(i)
                used += p_arr[i]
            else:
                late.append(i)
        # Already in non-increasing p/alpha order for `early`.
        # Sort `late` by non-decreasing p_i / beta_i.
        def late_key(i):
            b = b_arr[i]
            if b <= 0:
                return (float("inf"), i)
            return (p_arr[i] / b, i)
        late.sort(key=late_key)
        perm = early + late
        return [i + 1 for i in perm]

    # ==================================================================
    # (4) Improvement
    # ==================================================================
    def _penalty(perm: Sequence[int]) -> int:
        c = 0
        total = 0
        for idx in perm:
            p, a, b = jobs[idx - 1]
            c += p
            if c < d_val:
                total += a * (d_val - c)
            elif c > d_val:
                total += b * (c - d_val)
        return total

    def apply_swap_2opt(perm: Sequence[int],
                        time_limit_s: float = 5.0) -> list:
        """Adjacent-pair swap local search with first-improvement.
        Repeatedly scan i=0..n-2, swap perm[i] and perm[i+1], keep the swap
        iff total penalty strictly decreases. Restarts from i=0 after each
        accepted move. Adjacent swaps are an O(1) delta in principle but we
        recompute the full penalty for safety (O(n) per move).

        Returns a NEW list. Self-monitors time_limit_s. Input must be a valid
        1-based permutation."""
        if len(perm) != n or sorted(perm) != list(range(1, n + 1)):
            raise ValueError("perm must be a permutation of 1..n")
        cur = list(perm)
        cur_cost = _penalty(cur)
        t0 = time.time()
        safety = 0.05
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            for i in range(n - 1):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                cur[i], cur[i + 1] = cur[i + 1], cur[i]
                new_cost = _penalty(cur)
                if new_cost < cur_cost:
                    cur_cost = new_cost
                    improved = True
                    break  # first-improvement restart
                else:
                    cur[i], cur[i + 1] = cur[i + 1], cur[i]  # undo
        return cur

    def apply_insertion_search(perm: Sequence[int],
                               time_limit_s: float = 5.0) -> list:
        """Insertion (or-opt-1) local search: for each position i, try
        removing perm[i] and reinserting it at every other position k,
        accept the move if total penalty strictly decreases.
        First-improvement; restarts from i=0 after each accepted move.
        Often escapes local optima that adjacent-swap is stuck in (e.g.
        moves across the due-date boundary).

        Returns a NEW list. Self-monitors time_limit_s. O(n^2) per pass with
        an O(n) recompute per candidate move (=> O(n^3) per pass)."""
        if len(perm) != n or sorted(perm) != list(range(1, n + 1)):
            raise ValueError("perm must be a permutation of 1..n")
        cur = list(perm)
        cur_cost = _penalty(cur)
        t0 = time.time()
        safety = 0.05
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            for i in range(n):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                x = cur[i]
                rest = cur[:i] + cur[i + 1:]
                best_k = None
                best_cost = cur_cost
                for k in range(n):
                    if k == i:
                        continue
                    candidate = rest[:k] + [x] + rest[k:]
                    c = _penalty(candidate)
                    if c < best_cost:
                        best_cost = c
                        best_k = k
                if best_k is not None:
                    cur = rest[:best_k] + [x] + rest[best_k:]
                    cur_cost = best_cost
                    improved = True
                    break
        return cur

    # ==================================================================
    # (5) Heavy: exact ILP
    # ==================================================================
    def ilp_cdd(time_limit_s: float = 30.0) -> Optional[list]:
        """Solve the common-due-date problem to optimality (or best-found
        within time_limit_s) via python-mip / CBC. Returns a 1-based
        permutation as list[int], or None if no feasible solution was found.

        Formulation (position-assignment ILP):
          x[j, k] = 1 iff job j (0-indexed) is placed at position k.
          C[k]    = completion time at position k.
          E[k]    = max(0, d - C[k])    (earliness at position k)
          T[k]    = max(0, C[k] - d)    (tardiness at position k)
          alpha[k] = sum_j a[j] * x[j,k], beta[k] = sum_j b[j] * x[j,k]
          Penalty contribution at position k = alpha[k]*E[k] + beta[k]*T[k]

        Since alpha[k]*E[k] is bilinear in x and E we can't model it as a
        plain MILP without linearization, so we instead choose the SEQUENCE
        formulation: decide pair-order y[i,j] = 1 iff i precedes j, with
        completion times derived from y. This is O(n^2) variables and is
        practical up to n ~ 25-30. For larger n the model is built but the
        solver likely times out -- returning whatever feasible solution CBC
        has found at the time limit (or None).
        """
        if n == 0:
            return []
        try:
            from mip import Model, BINARY, CONTINUOUS, MINIMIZE, xsum, OptimizationStatus
        except ImportError:
            return None

        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)

        # y[i][j] = 1 iff i precedes j (i != j). Symmetry: y[i,j] + y[j,i] = 1.
        y = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    y[i][j] = m.add_var(var_type=BINARY, name=f"y_{i}_{j}")
        # C[i] = completion time of job i.
        C = [m.add_var(var_type=CONTINUOUS, lb=0.0, name=f"C_{i}") for i in range(n)]
        # E[i] = earliness, T[i] = tardiness (both >= 0).
        E = [m.add_var(var_type=CONTINUOUS, lb=0.0, name=f"E_{i}") for i in range(n)]
        T = [m.add_var(var_type=CONTINUOUS, lb=0.0, name=f"T_{i}") for i in range(n)]

        # Order symmetry.
        for i in range(n):
            for j in range(i + 1, n):
                m += y[i][j] + y[j][i] == 1, f"sym_{i}_{j}"

        # Completion time: C_j = p_j + sum_{i != j} p_i * y[i][j].
        for j in range(n):
            m += C[j] == p_arr[j] + xsum(p_arr[i] * y[i][j]
                                          for i in range(n) if i != j), f"C_def_{j}"

        # Earliness / tardiness linearization (both bounded below).
        for j in range(n):
            m += E[j] >= d_val - C[j], f"E_def_{j}"
            m += T[j] >= C[j] - d_val, f"T_def_{j}"

        m.objective = xsum(a_arr[j] * E[j] + b_arr[j] * T[j] for j in range(n))

        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None

        # Recover permutation by sorting jobs on completion time.
        completions = []
        for j in range(n):
            v = C[j].x
            if v is None:
                return None
            completions.append((float(v), j))
        completions.sort()
        return [j + 1 for (_c, j) in completions]

    return {
        "processing_time": processing_time,
        "penalty_early": penalty_early,
        "penalty_late": penalty_late,
        "due_date": due_date,
        "n_jobs": n_jobs,
        "compute_total_penalty": compute_total_penalty,
        "edd_construct": edd_construct,
        "wspt_construct": wspt_construct,
        "v_shape_construct": v_shape_construct,
        "apply_swap_2opt": apply_swap_2opt,
        "apply_insertion_search": apply_insertion_search,
        "ilp_cdd": ilp_cdd,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "processing_time",
        "input": "j: int",
        "output": "int",
        "purpose": "Processing time p_j of job j (1-based index).",
    },
    {
        "name": "penalty_early",
        "input": "j: int",
        "output": "int",
        "purpose": "Earliness penalty coefficient alpha_j of job j (1-based).",
    },
    {
        "name": "penalty_late",
        "input": "j: int",
        "output": "int",
        "purpose": "Tardiness penalty coefficient beta_j of job j (1-based).",
    },
    {
        "name": "due_date",
        "input": "(no args)",
        "output": "int",
        "purpose": (
            "Common due date d = floor(sum(p) * h). Same value used by "
            "CO-Bench eval_func, so you can compare cumulative completion "
            "times against this without recomputing."
        ),
    },
    {
        "name": "n_jobs",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of jobs n in this instance.",
    },
    # ----- Feasibility primitive -----
    {
        "name": "compute_total_penalty",
        "input": "permutation: Sequence[int], start_time: int = 0",
        "output": "int",
        "purpose": (
            "Total earliness+tardiness penalty for processing jobs in "
            "`permutation` (1-based) order starting at `start_time`. Matches "
            "CO-Bench eval_func when start_time=0. Faster than tools["
            "'objective'] (skips the feasibility wrapping). Use inside local "
            "search to evaluate candidate permutations."
        ),
    },
    # ----- Construction -----
    {
        "name": "edd_construct",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Construction heuristic: with a common due date, EDD degenerates "
            "to SPT (sort by non-decreasing p_i). Returns a 1-based "
            "permutation. Cheap warm start."
        ),
    },
    {
        "name": "wspt_construct",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Weighted Shortest Processing Time: sort jobs by ascending "
            "p_i / max(beta_i, 1). Returns a 1-based permutation. Useful "
            "when most jobs end up tardy (small h or large p)."
        ),
    },
    {
        "name": "v_shape_construct",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Panwalkar-Smith V-shape construction. Greedily assigns jobs to "
            "an early set (filled in decreasing p_i/alpha_i until sum of p "
            "exceeds d) and a late set (the rest). Early jobs are kept in "
            "non-increasing p_i/alpha_i order; late jobs are sorted by "
            "non-decreasing p_i/beta_i. Returns a 1-based permutation. "
            "Strong starting point: any optimum has the V-shape, though the "
            "exact split point may need local-search refinement."
        ),
    },
    # ----- Improvement -----
    {
        "name": "apply_swap_2opt",
        "input": "perm: Sequence[int], time_limit_s: float = 5.0",
        "output": "list[int]",
        "purpose": (
            "Adjacent-pair swap local search with first-improvement. Useful "
            "to refine v_shape_construct / wspt_construct / edd_construct. "
            "Returns a NEW 1-based permutation. Self-monitors time_limit_s."
        ),
    },
    {
        "name": "apply_insertion_search",
        "input": "perm: Sequence[int], time_limit_s: float = 5.0",
        "output": "list[int]",
        "purpose": (
            "Or-opt-1 (single-job reinsertion) local search with "
            "first-improvement. Each pass tries moving every job to every "
            "other position; complementary to apply_swap_2opt because it "
            "can shift jobs across the due-date boundary in one move. "
            "Returns a NEW 1-based permutation. Self-monitors time_limit_s."
        ),
    },
    # ----- Heavy -----
    {
        "name": "ilp_cdd",
        "input": "time_limit_s: float = 30.0",
        "output": "list[int] | None",
        "purpose": (
            "Solve the common-due-date problem exactly via python-mip / CBC "
            "with a pair-order (y[i,j]) MILP formulation. Returns a 1-based "
            "permutation or None if no feasible solution was found inside "
            "the time limit. Practical up to n ~ 25-30; for larger n the "
            "solver likely times out (returning either CBC's best feasible "
            "incumbent or None). Combine with v_shape_construct on larger "
            "instances where ILP alone is too slow."
        ),
    },
]
