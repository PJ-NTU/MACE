"""Per-problem extras for CO-Bench Open Shop Scheduling.

Open Shop differs from JSSP / Flow Shop: there is NO precedence between the
operations of a single job. Every job has one operation on each machine and
those operations can be executed in any order, subject only to:
  (i)  a job is processed on at most one machine at a time, and
  (ii) a machine processes at most one job at a time.

Solution schema: {"start_times": list[list[int]]} with shape n_jobs x n_machines,
where start_times[j][op] is the start time of the op-th operation of job j --
the SAME op-index that indexes times[j] and machines[j] in the instance. The
machine that op runs on is machines[j][op] (1-indexed in CO-Bench).

Tool groups:
  (1) Queries:        processing_time, n_jobs, n_machines
  (2) Feasibility:    simulate_makespan_from_starts, job_completion,
                      machine_completion, validate_partial
  (3) Construction:   lpt_dense_construct, greedy_list_schedule
  (4) Improvement:    apply_local_swap
  (5) Heavy:          ilp_open_shop
"""
from __future__ import annotations
import time
from collections import defaultdict
from typing import Optional


def extra_tools(instance: dict) -> dict:
    """Factory: returns OSSP-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench Open shop load_data):
      - n_jobs:     int
      - n_machines: int  (also = operations per job)
      - times:      list[list[int]], shape n_jobs x n_machines
                    times[j][op] = processing time of job j's op-th operation
      - machines:   list[list[int]], shape n_jobs x n_machines (1-indexed)
                    machines[j][op] = machine id for job j's op-th operation
      - upper_bound, lower_bound: ints (optional reference values)
    """
    nj = int(instance["n_jobs"])
    nm = int(instance["n_machines"])
    times = [list(row) for row in instance["times"]]
    machines = [list(row) for row in instance["machines"]]

    if len(times) != nj or len(machines) != nj:
        raise ValueError(f"times/machines must have {nj} rows")
    for i in range(nj):
        if len(times[i]) != nm or len(machines[i]) != nm:
            raise ValueError(f"row {i} must have {nm} entries")

    # Distinct machine ids actually used (1-indexed convention from CO-Bench).
    machine_ids = sorted({machines[i][j] for i in range(nj) for j in range(nm)})

    # (job, machine_id) -> op_idx within job j (i.e., column in times/machines)
    job_machine_to_op: dict[tuple[int, int], int] = {}
    # machine_id -> list of (job, op_idx) using that machine
    machine_to_ops: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for j in range(nj):
        for op in range(nm):
            mid = machines[j][op]
            job_machine_to_op[(j, mid)] = op
            machine_to_ops[mid].append((j, op))

    # ==================================================================
    # (1) Queries -- O(1)
    # ==================================================================
    def processing_time(j: int, m: int) -> int:
        """Processing time of job j on machine m (m is the 1-indexed machine
        id matching instance['machines'])."""
        key = (int(j), int(m))
        if key not in job_machine_to_op:
            raise ValueError(
                f"no operation for (job={j}, machine={m}); "
                f"valid jobs [0,{nj}), machine ids {machine_ids}"
            )
        return int(times[j][job_machine_to_op[key]])

    def n_jobs() -> int:
        return nj

    def n_machines() -> int:
        return nm

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _check_shape(start_times) -> Optional[str]:
        if not isinstance(start_times, list) or len(start_times) != nj:
            return f"start_times must be list of {nj} rows"
        for i, row in enumerate(start_times):
            if not isinstance(row, list) or len(row) != nm:
                return f"row {i} must be list of {nm} ints"
        return None

    def simulate_makespan_from_starts(start_times) -> Optional[int]:
        """Compute the makespan implied by `start_times` IF feasible, else None.

        Checks:
          - shape (n_jobs x n_machines)
          - non-negativity
          - no two ops of the same job overlap in time
          - no two ops on the same machine overlap in time
        Cheaper than calling tools['objective'] when you just want a yes/no +
        the makespan value without the framework's eval_func round-trip.
        O(n_jobs * n_machines * log(n_machines))."""
        if _check_shape(start_times) is not None:
            return None
        for i in range(nj):
            for j in range(nm):
                if start_times[i][j] < 0:
                    return None
        # Per-job no overlap.
        for i in range(nj):
            intervals = sorted(
                ((start_times[i][op], start_times[i][op] + times[i][op])
                 for op in range(nm)),
                key=lambda x: x[0],
            )
            for k in range(1, nm):
                if intervals[k - 1][1] > intervals[k][0]:
                    return None
        # Per-machine no overlap.
        for mid, ops in machine_to_ops.items():
            intervals = sorted(
                ((start_times[i][o], start_times[i][o] + times[i][o])
                 for (i, o) in ops),
                key=lambda x: x[0],
            )
            for k in range(1, len(intervals)):
                if intervals[k - 1][1] > intervals[k][0]:
                    return None
        # Makespan = max end across all ops.
        ms = 0
        for i in range(nj):
            for op in range(nm):
                end = start_times[i][op] + times[i][op]
                if end > ms:
                    ms = end
        return int(ms)

    def job_completion(j: int, start_times) -> int:
        """Latest end-time over job j's ops in `start_times`. No feasibility
        check; if you pass an infeasible schedule the number is meaningless."""
        if not (0 <= j < nj):
            raise ValueError(f"j={j} out of range for {nj} jobs")
        err = _check_shape(start_times)
        if err is not None:
            raise ValueError(err)
        best = 0
        for op in range(nm):
            end = start_times[j][op] + times[j][op]
            if end > best:
                best = end
        return int(best)

    def machine_completion(m: int, start_times) -> int:
        """Latest end-time on machine m under `start_times`. No feasibility
        check. Returns 0 if no ops on that machine."""
        err = _check_shape(start_times)
        if err is not None:
            raise ValueError(err)
        ops = machine_to_ops.get(int(m), [])
        best = 0
        for (i, op) in ops:
            end = start_times[i][op] + times[i][op]
            if end > best:
                best = end
        return int(best)

    def validate_partial(start_times) -> list[str]:
        """Return all violated constraints (empty list => feasible). Reports
        shape, non-negativity, per-job overlap, and per-machine overlap. More
        informative than is_feasible for debugging local-search moves."""
        problems: list[str] = []
        err = _check_shape(start_times)
        if err is not None:
            return [err]
        for i in range(nj):
            for op in range(nm):
                if start_times[i][op] < 0:
                    problems.append(
                        f"start_times[{i}][{op}]={start_times[i][op]} < 0"
                    )
        for i in range(nj):
            intervals = sorted(
                ((start_times[i][op], start_times[i][op] + times[i][op], op)
                 for op in range(nm)),
                key=lambda x: x[0],
            )
            for k in range(1, nm):
                ps, pe, po = intervals[k - 1]
                cs, ce, co = intervals[k]
                if pe > cs:
                    problems.append(
                        f"job {i}: op {po} (ends {pe}) overlaps op {co} (starts {cs})"
                    )
        for mid, ops in machine_to_ops.items():
            intervals = sorted(
                ((start_times[i][o], start_times[i][o] + times[i][o], i, o)
                 for (i, o) in ops),
                key=lambda x: x[0],
            )
            for k in range(1, len(intervals)):
                ps, pe, pi, po = intervals[k - 1]
                cs, ce, ci, co = intervals[k]
                if pe > cs:
                    problems.append(
                        f"machine {mid}: job {pi} op {po} (ends {pe}) overlaps "
                        f"job {ci} op {co} (starts {cs})"
                    )
        return problems

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def _schedule_op(j: int, op: int, mid: int,
                     job_ready: list, machine_ready: dict,
                     start_times: list) -> int:
        st = max(job_ready[j], machine_ready[mid])
        pt = times[j][op]
        start_times[j][op] = st
        end = st + pt
        job_ready[j] = end
        machine_ready[mid] = end
        return end

    def lpt_dense_construct() -> list[list[int]]:
        """Longest-Processing-Time dense construction.

        At each step, consider the set of UNSCHEDULED operations whose job and
        machine are BOTH currently free (or can become free without idle gap
        relative to the global clock). Pick the one with the longest processing
        time; ties broken by (job, machine_id). Repeat until all n_jobs*n_machines
        ops are scheduled. Always produces a feasible schedule.

        This is the open-shop analogue of LPT: longest-first packing tends to
        spread the bottleneck work early, reducing makespan tail. Greedy and
        O((n_jobs * n_machines)^2)."""
        start_times = [[0] * nm for _ in range(nj)]
        scheduled = [[False] * nm for _ in range(nj)]
        job_ready = [0] * nj
        machine_ready = {mid: 0 for mid in machine_ids}
        remaining = nj * nm
        while remaining > 0:
            # Earliest time at which SOME (job, op) can start: min over all
            # unscheduled ops of max(job_ready[j], machine_ready[mid]).
            now = None
            for j in range(nj):
                for op in range(nm):
                    if scheduled[j][op]:
                        continue
                    t_start = max(job_ready[j], machine_ready[machines[j][op]])
                    if now is None or t_start < now:
                        now = t_start
            # Among ops whose earliest start == now, pick the one with the
            # LONGEST processing time.
            best = None  # (-pt, j, op, mid)
            for j in range(nj):
                for op in range(nm):
                    if scheduled[j][op]:
                        continue
                    mid = machines[j][op]
                    t_start = max(job_ready[j], machine_ready[mid])
                    if t_start != now:
                        continue
                    key = (-int(times[j][op]), j, mid)
                    if best is None or key < best[0]:
                        best = (key, j, op, mid)
            _, j, op, mid = best
            _schedule_op(j, op, mid, job_ready, machine_ready, start_times)
            scheduled[j][op] = True
            remaining -= 1
        return start_times

    def greedy_list_schedule(priorities: Optional[list] = None) -> list[list[int]]:
        """List scheduling driven by a priority list of (job, machine_id) pairs.

        `priorities` is a flat list of (j, m) pairs covering EVERY operation
        exactly once. We scan it left-to-right; each op is placed at the
        earliest feasible time given previously placed ops (i.e.,
        max(job_ready[j], machine_ready[m])). Always feasible.

        If `priorities` is None we use a default ordering: ops sorted by
        descending processing time (a 'list-LPT' rule), ties by (job, machine).
        O((n_jobs * n_machines))."""
        if priorities is None:
            order = []
            for j in range(nj):
                for op in range(nm):
                    order.append((-int(times[j][op]), j, machines[j][op], op))
            order.sort()
            priorities_resolved = [(j, mid) for (_, j, mid, _) in order]
        else:
            priorities_resolved = list(priorities)
        # Validate.
        seen: set[tuple[int, int]] = set()
        for pair in priorities_resolved:
            if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                raise ValueError(f"priorities entry must be (job, machine): {pair}")
            jj, mm = int(pair[0]), int(pair[1])
            if (jj, mm) not in job_machine_to_op:
                raise ValueError(f"no operation for (job={jj}, machine={mm})")
            if (jj, mm) in seen:
                raise ValueError(f"duplicate (job={jj}, machine={mm}) in priorities")
            seen.add((jj, mm))
        if len(seen) != nj * nm:
            raise ValueError(
                f"priorities must cover all {nj * nm} ops; got {len(seen)}"
            )
        start_times = [[0] * nm for _ in range(nj)]
        job_ready = [0] * nj
        machine_ready = {mid: 0 for mid in machine_ids}
        for (j, mid) in priorities_resolved:
            op = job_machine_to_op[(j, mid)]
            _schedule_op(j, op, mid, job_ready, machine_ready, start_times)
        return start_times

    # ==================================================================
    # (4) Improvement
    # ==================================================================
    def _extract_priority_from_starts(start_times) -> list[tuple[int, int]]:
        """Order all ops by (start, job, machine_id) -- a left-to-right scan of
        the Gantt chart. Right-shifting a feasible schedule preserves this
        order under greedy_list_schedule, so this is the canonical 'priority
        list' for the given schedule."""
        flat = []
        for j in range(nj):
            for op in range(nm):
                flat.append((start_times[j][op], j, machines[j][op]))
        flat.sort()
        return [(j, mid) for (_, j, mid) in flat]

    def _makespan_lb(start_times) -> int:
        ms = 0
        for j in range(nj):
            for op in range(nm):
                end = start_times[j][op] + times[j][op]
                if end > ms:
                    ms = end
        return ms

    def apply_local_swap(start_times,
                         time_limit_s: float = 5.0) -> list[list[int]]:
        """Local search: extract the priority list induced by `start_times`,
        try swapping each adjacent pair, re-list-schedule, and keep the swap
        iff the makespan strictly decreases. First-improvement. Restarts after
        each accepted move. Self-monitors time_limit_s. Returns a NEW
        start_times (does not mutate input).

        Caller must pass a feasible warm start (lpt_dense_construct or
        greedy_list_schedule both qualify). Open-shop neighborhood:
        O(n_jobs * n_machines) candidate swaps per pass, each re-simulation is
        O(n_jobs * n_machines).
        """
        err = _check_shape(start_times)
        if err is not None:
            raise ValueError(err)
        ms_now = simulate_makespan_from_starts(start_times)
        if ms_now is None:
            raise ValueError("input start_times is not feasible")
        priority = _extract_priority_from_starts(start_times)
        cur = [list(row) for row in start_times]
        cur_ms = ms_now
        t0 = time.time()
        safety = 0.05
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            for k in range(len(priority) - 1):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                new_pri = list(priority)
                new_pri[k], new_pri[k + 1] = new_pri[k + 1], new_pri[k]
                try:
                    cand = greedy_list_schedule(new_pri)
                except ValueError:
                    continue
                cand_ms = _makespan_lb(cand)
                if cand_ms < cur_ms:
                    cur = cand
                    cur_ms = cand_ms
                    priority = _extract_priority_from_starts(cur)
                    improved = True
                    break
        return cur

    # ==================================================================
    # (5) Heavy: exact ILP via python-mip / CBC
    # ==================================================================
    def ilp_open_shop(time_limit_s: float = 30.0) -> Optional[list[list[int]]]:
        """Solve the Open Shop disjunctive ILP exactly with python-mip / CBC.

        Returns optimal-or-best-found start_times within time_limit_s, or None
        if no feasible solution. For small instances (n_jobs, n_machines <= 7)
        this typically finds the optimum. Taillard 10x10+ usually times out
        with a feasible-but-suboptimal solution -- still a useful warm start.

        Decision variables:
          s[j][op] >= 0      : start of op
          Cmax              : makespan, minimized
          y_a_b             : binary, 1 iff op-a precedes op-b on the same
                              MACHINE (one var per unordered pair of ops
                              sharing a machine)
          z_a_b             : binary, 1 iff op-a precedes op-b within the same
                              JOB (one var per unordered pair of ops in same job)
        Big-M = sum of all processing times + 1.
        """
        try:
            from mip import (Model, INTEGER, BINARY, MINIMIZE, xsum,
                             OptimizationStatus)
        except ImportError:
            return None
        total_p = sum(int(times[i][j]) for i in range(nj) for j in range(nm))
        BIGM = total_p + 1
        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)
        s = [[m.add_var(var_type=INTEGER, lb=0, ub=BIGM, name=f"s_{i}_{j}")
              for j in range(nm)] for i in range(nj)]
        Cmax = m.add_var(var_type=INTEGER, lb=0, ub=BIGM, name="Cmax")
        m.objective = Cmax
        for i in range(nj):
            for j in range(nm):
                m += Cmax >= s[i][j] + int(times[i][j])
        # Per-machine disjunctive constraints.
        for mid in machine_ids:
            ops = machine_to_ops[mid]
            for a in range(len(ops)):
                ia, ja = ops[a]
                for b in range(a + 1, len(ops)):
                    ib, jb = ops[b]
                    y = m.add_var(var_type=BINARY,
                                  name=f"ym_{ia}_{ja}_{ib}_{jb}")
                    m += s[ia][ja] + int(times[ia][ja]) <= s[ib][jb] + BIGM * (1 - y)
                    m += s[ib][jb] + int(times[ib][jb]) <= s[ia][ja] + BIGM * y
        # Per-job disjunctive constraints (open shop has NO fixed order).
        for i in range(nj):
            for a in range(nm):
                for b in range(a + 1, nm):
                    z = m.add_var(var_type=BINARY, name=f"yj_{i}_{a}_{b}")
                    m += s[i][a] + int(times[i][a]) <= s[i][b] + BIGM * (1 - z)
                    m += s[i][b] + int(times[i][b]) <= s[i][a] + BIGM * z
        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL,
                          OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None
        out = [[int(round(s[i][j].x)) for j in range(nm)] for i in range(nj)]
        return out

    return {
        "processing_time": processing_time,
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "simulate_makespan_from_starts": simulate_makespan_from_starts,
        "job_completion": job_completion,
        "machine_completion": machine_completion,
        "validate_partial": validate_partial,
        "lpt_dense_construct": lpt_dense_construct,
        "greedy_list_schedule": greedy_list_schedule,
        "apply_local_swap": apply_local_swap,
        "ilp_open_shop": ilp_open_shop,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "processing_time",
        "input": "j: int, m: int",
        "output": "int",
        "purpose": (
            "Processing time of job j's operation on machine m (m is the "
            "1-indexed machine id matching instance['machines']). Raises if "
            "(j, m) is not a valid operation. O(1)."
        ),
    },
    {
        "name": "n_jobs",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of jobs in the instance.",
    },
    {
        "name": "n_machines",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of machines (also = ops per job).",
    },
    # ----- Feasibility primitives -----
    {
        "name": "simulate_makespan_from_starts",
        "input": "start_times: list[list[int]]",
        "output": "int | None",
        "purpose": (
            "If `start_times` is feasible (shape ok, non-negative, no per-job "
            "or per-machine overlap), return its makespan; else return None. "
            "Cheaper than tools['objective'] when you only need the makespan "
            "and feasibility check. O(n_jobs * n_machines * log(n_machines))."
        ),
    },
    {
        "name": "job_completion",
        "input": "j: int, start_times: list[list[int]]",
        "output": "int",
        "purpose": (
            "Latest end-time across job j's ops in `start_times`. No "
            "feasibility check. O(n_machines)."
        ),
    },
    {
        "name": "machine_completion",
        "input": "m: int, start_times: list[list[int]]",
        "output": "int",
        "purpose": (
            "Latest end-time on machine m under `start_times` (earliest moment "
            "the machine becomes globally free). No feasibility check. "
            "Returns 0 if no ops on that machine. O(n_jobs)."
        ),
    },
    {
        "name": "validate_partial",
        "input": "start_times: list[list[int]]",
        "output": "list[str]",
        "purpose": (
            "Returns every violated constraint (shape, non-negativity, per-job "
            "overlap, per-machine overlap) as a list of messages. Empty list "
            "means feasible. More informative than is_feasible's single-error "
            "result -- use inside local search to see ALL violations at once."
        ),
    },
    # ----- Construction -----
    {
        "name": "lpt_dense_construct",
        "input": "(no args)",
        "output": "list[list[int]] (start_times)",
        "purpose": (
            "LPT-style dense construction. Repeatedly: among ops whose (job, "
            "machine) can both start at the current earliest time, pick the "
            "one with the LONGEST processing time. Always feasible. Tends to "
            "front-load bottleneck work, shrinking the tail. Good warm start. "
            "O((n_jobs * n_machines)^2)."
        ),
    },
    {
        "name": "greedy_list_schedule",
        "input": "priorities: list[(job, machine)] | None = None",
        "output": "list[list[int]] (start_times)",
        "purpose": (
            "List scheduling driven by a priority list of (job, machine_id) "
            "pairs covering every op exactly once. Each op placed at the "
            "earliest feasible time given previously placed ops. Always "
            "feasible. Pass None for a default LPT-by-processing-time ordering. "
            "O(n_jobs * n_machines)."
        ),
    },
    # ----- Improvement -----
    {
        "name": "apply_local_swap",
        "input": "start_times: list[list[int]], time_limit_s: float = 5.0",
        "output": "list[list[int]] (start_times)",
        "purpose": (
            "Local search in priority-list space: extract the (start, job, "
            "machine)-sorted priority list of `start_times`, try every "
            "adjacent-pair swap, re-list-schedule, and accept iff the new "
            "makespan strictly decreases. First-improvement; restarts after "
            "each accepted move. Caller must supply a feasible warm start "
            "(e.g., from lpt_dense_construct). Returns a NEW start_times."
        ),
    },
    # ----- Heavy: exact ILP -----
    {
        "name": "ilp_open_shop",
        "input": "time_limit_s: float = 30.0",
        "output": "list[list[int]] | None (start_times)",
        "purpose": (
            "Solve the Open Shop disjunctive ILP exactly with python-mip / "
            "CBC. Decision: start of each op + disjunctive binaries on every "
            "pair of ops sharing a machine AND on every pair of ops within "
            "the same job (open shop has no fixed job order). Returns "
            "optimal-or-best-found start_times within time_limit_s, or None if "
            "no feasible solution. Effective on small instances (<= ~7x7); "
            "Taillard 10x10+ usually times out with a feasible suboptimal "
            "solution -- still a strong warm start for local search."
        ),
    },
]
