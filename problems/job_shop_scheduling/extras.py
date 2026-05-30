"""Per-problem extras for CO-Bench Job Shop Scheduling.

Provides building blocks so the LLM can compose dispatching + local-search
heuristics for JSSP without reinventing simulation, critical-path swaps, etc.

Solution schema: {"start_times": list[list[int]]} with shape n_jobs x n_machines.
Machine ids in `instance["machines"]` are 1-indexed (matches CO-Bench).

Tool groups:
  (1) Queries:        processing_time, machine_of, total_work, n_jobs, n_machines
  (2) Feasibility:    validate_partial, job_completion_time, machine_load
  (3) Construction:   spt_dispatch, simulate_active_schedule
  (4) Improvement:    apply_critical_path_swap
  (5) Heavy:          ilp_jssp
"""
from __future__ import annotations
import time
from collections import defaultdict
from typing import Optional


def extra_tools(instance: dict) -> dict:
    """Factory: returns JSSP-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench JSSP load_data):
      - n_jobs:     int
      - n_machines: int  (also = operations per job)
      - times:      list[list[int]], shape n_jobs x n_machines
      - machines:   list[list[int]], shape n_jobs x n_machines (1-indexed)
      - upper_bound, lower_bound: ints (optional reference values)
    """
    nj = int(instance["n_jobs"])
    nm = int(instance["n_machines"])
    times = [list(row) for row in instance["times"]]
    machines = [list(row) for row in instance["machines"]]

    # Validate shapes once.
    if len(times) != nj or len(machines) != nj:
        raise ValueError(f"times/machines must have {nj} rows")
    for i in range(nj):
        if len(times[i]) != nm or len(machines[i]) != nm:
            raise ValueError(f"row {i} must have {nm} entries")

    # Distinct machine ids actually used (preserves 1-indexed convention).
    machine_ids = sorted({machines[i][j] for i in range(nj) for j in range(nm)})

    # Precompute inverse map: machine_id -> list of (job, op_idx) using it.
    machine_to_ops: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i in range(nj):
        for j in range(nm):
            machine_to_ops[machines[i][j]].append((i, j))

    # ==================================================================
    # (1) Queries -- O(1)
    # ==================================================================
    def processing_time(j: int, op: int) -> int:
        if not (0 <= j < nj and 0 <= op < nm):
            raise ValueError(f"(j={j}, op={op}) out of range for {nj}x{nm}")
        return int(times[j][op])

    def machine_of(j: int, op: int) -> int:
        if not (0 <= j < nj and 0 <= op < nm):
            raise ValueError(f"(j={j}, op={op}) out of range for {nj}x{nm}")
        return int(machines[j][op])

    def total_work(j: int) -> int:
        if not (0 <= j < nj):
            raise ValueError(f"j={j} out of range for {nj} jobs")
        return int(sum(times[j]))

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

    def validate_partial(start_times) -> list[str]:
        """Return list of violation messages (empty list => feasible).
        Checks: shape, non-negativity, per-job precedence, per-machine overlap.
        Useful inside local search to enumerate every violated constraint at
        once instead of bisecting via is_feasible's single-error message.
        O(n_jobs * n_machines + per-machine sort)."""
        problems: list[str] = []
        shape_err = _check_shape(start_times)
        if shape_err is not None:
            return [shape_err]
        # Non-negativity.
        for i in range(nj):
            for j in range(nm):
                if start_times[i][j] < 0:
                    problems.append(f"start_times[{i}][{j}]={start_times[i][j]} < 0")
        # Precedence.
        for i in range(nj):
            for j in range(1, nm):
                prev_end = start_times[i][j - 1] + times[i][j - 1]
                if start_times[i][j] < prev_end:
                    problems.append(
                        f"job {i} op {j} starts {start_times[i][j]} < prev end {prev_end}"
                    )
        # Machine overlap.
        for mid, ops in machine_to_ops.items():
            intervals = sorted(
                ((start_times[i][j], start_times[i][j] + times[i][j], i, j)
                 for (i, j) in ops),
                key=lambda x: x[0],
            )
            for k in range(1, len(intervals)):
                ps, pe, pi, pj = intervals[k - 1]
                cs, ce, ci, cj = intervals[k]
                if pe > cs:
                    problems.append(
                        f"machine {mid}: job {pi} op {pj} (ends {pe}) overlaps "
                        f"job {ci} op {cj} (starts {cs})"
                    )
        return problems

    def job_completion_time(j: int, start_times) -> int:
        """Completion time of job j given start_times (no feasibility check).
        Returns start_times[j][last] + times[j][last]."""
        if not (0 <= j < nj):
            raise ValueError(f"j={j} out of range for {nj} jobs")
        shape_err = _check_shape(start_times)
        if shape_err is not None:
            raise ValueError(shape_err)
        return int(start_times[j][nm - 1] + times[j][nm - 1])

    def machine_load(m: int, start_times) -> int:
        """Current end-of-machine time on machine `m` under start_times.
        i.e., max(start + processing_time) over ops assigned to m.
        Useful for greedy: 'next op on m can't start before machine_load(m,...)'.
        Returns 0 if no ops on that machine."""
        shape_err = _check_shape(start_times)
        if shape_err is not None:
            raise ValueError(shape_err)
        ops = machine_to_ops.get(int(m), [])
        best = 0
        for (i, j) in ops:
            end = start_times[i][j] + times[i][j]
            if end > best:
                best = end
        return int(best)

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def simulate_active_schedule(machine_sequences: dict) -> list[list[int]]:
        """Given a dict {machine_id: [job_ids_in_order]} specifying the order
        each machine processes its operations, compute the implied start_times
        by event-driven simulation. Each op starts at max(job-ready, machine-
        ready). Returns a fresh 2D list (NOT a mutation of any caller state).

        Raises ValueError if the sequences are malformed (missing job, wrong
        length) or if the precedence creates a deadlock (cyclic wait).

        Bridge between sequence-style representations and the start_times
        solution format. O((n_jobs * n_machines)^2) worst case but usually
        linear-ish in practice.
        """
        if not isinstance(machine_sequences, dict):
            raise ValueError("machine_sequences must be dict {machine: [jobs]}")
        # Validate: each machine's seq is a permutation of the jobs that use it.
        seq_iter = {}  # machine -> list of (job, op_idx) in chosen order
        for mid in machine_ids:
            expected = machine_to_ops[mid]
            expected_jobs = [j for (j, _) in expected]
            given = machine_sequences.get(mid)
            if given is None:
                raise ValueError(f"missing sequence for machine {mid}")
            if sorted(given) != sorted(expected_jobs):
                raise ValueError(
                    f"machine {mid}: sequence {given} != expected jobs {sorted(expected_jobs)}"
                )
            # For each occurrence of job j in `given`, attach next unused op_idx
            # (since each job uses each machine exactly once in standard JSSP,
            # this is a 1-1 mapping; but be robust).
            job_ops = defaultdict(list)
            for (j, op_idx) in expected:
                job_ops[j].append(op_idx)
            ordered = []
            for j in given:
                ordered.append((j, job_ops[j].pop(0)))
            seq_iter[mid] = ordered

        # Initialize start_times. Track per-job next-op index and per-machine
        # next-position index. Repeat until all ops scheduled.
        start_times = [[0] * nm for _ in range(nj)]
        scheduled = [[False] * nm for _ in range(nj)]
        job_ready = [0] * nj      # earliest time job j's next op can start
        next_job_op = [0] * nj    # next op index per job
        machine_ready = {mid: 0 for mid in machine_ids}
        machine_pos = {mid: 0 for mid in machine_ids}  # cursor into seq

        total_ops = nj * nm
        done = 0
        # Each pass scans every machine for a ready front-of-queue op.
        while done < total_ops:
            progressed = False
            for mid in machine_ids:
                pos = machine_pos[mid]
                seq = seq_iter[mid]
                if pos >= len(seq):
                    continue
                (j, op_idx) = seq[pos]
                # Is this the job's next op?
                if next_job_op[j] != op_idx:
                    continue
                st = max(job_ready[j], machine_ready[mid])
                start_times[j][op_idx] = st
                scheduled[j][op_idx] = True
                end = st + times[j][op_idx]
                job_ready[j] = end
                next_job_op[j] = op_idx + 1
                machine_ready[mid] = end
                machine_pos[mid] = pos + 1
                done += 1
                progressed = True
            if not progressed:
                raise ValueError(
                    "machine_sequences imply a deadlock (cyclic precedence)"
                )
        return start_times

    def spt_dispatch() -> list[list[int]]:
        """Shortest-Processing-Time dispatch: simulate building the schedule
        operation-by-operation. At each step, among all jobs whose next op is
        ready, pick the one with the smallest processing time. Ties broken by
        job index. Always produces a feasible (active) schedule.
        O((n_jobs * n_machines)^2) worst case. Good warm start."""
        start_times = [[0] * nm for _ in range(nj)]
        next_op = [0] * nj
        job_ready = [0] * nj
        machine_ready = {mid: 0 for mid in machine_ids}
        remaining = nj * nm
        while remaining > 0:
            # Eligible ops: job has remaining ops.
            best = None  # (proc_time, job_id, op_idx, mid, st)
            for j in range(nj):
                if next_op[j] >= nm:
                    continue
                op = next_op[j]
                mid = machines[j][op]
                st = max(job_ready[j], machine_ready[mid])
                pt = times[j][op]
                key = (pt, j)
                if best is None or key < best[0]:
                    best = (key, j, op, mid, st, pt)
            assert best is not None
            _, j, op, mid, st, pt = best
            start_times[j][op] = st
            end = st + pt
            job_ready[j] = end
            machine_ready[mid] = end
            next_op[j] += 1
            remaining -= 1
        return start_times

    # ==================================================================
    # (4) Improvement
    # ==================================================================
    def _makespan(start_times) -> int:
        return max(start_times[i][nm - 1] + times[i][nm - 1] for i in range(nj))

    def _extract_machine_sequences(start_times) -> dict:
        """From start_times, derive each machine's processing order by sorting
        its assigned ops by start time."""
        out = {}
        for mid in machine_ids:
            ops = machine_to_ops[mid]
            sorted_ops = sorted(ops, key=lambda jo: start_times[jo[0]][jo[1]])
            out[mid] = [j for (j, _) in sorted_ops]
        return out

    def apply_critical_path_swap(start_times,
                                 time_limit_s: float = 5.0) -> list[list[int]]:
        """Local search: repeatedly find one critical-path block (ops whose
        finish == makespan, tracing predecessors) on a single machine, swap an
        adjacent pair on that machine, re-simulate, and keep the swap if the
        makespan strictly decreases. First-improvement; restarts after each
        accepted move. Self-monitors time_limit_s. Returns a NEW start_times
        (does not mutate input). O((n_jobs * n_machines)^2) per pass.

        Use after spt_dispatch (or any feasible warm start) to refine. Caller
        is responsible for ensuring the input is feasible.
        """
        shape_err = _check_shape(start_times)
        if shape_err is not None:
            raise ValueError(shape_err)
        cur = [list(row) for row in start_times]
        cur_ms = _makespan(cur)
        t0 = time.time()
        safety = 0.05
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            # Find a critical operation: any (j, op) with end == makespan.
            critical_path: list[tuple[int, int]] = []
            target_j = None
            target_op = None
            for j in range(nj):
                if cur[j][nm - 1] + times[j][nm - 1] == cur_ms:
                    target_j = j
                    target_op = nm - 1
                    break
            if target_j is None:
                break
            # Trace backwards on the critical path: at each step, predecessor
            # is either the same-job prev op, or the same-machine prev op.
            cj, cop = target_j, target_op
            critical_path.append((cj, cop))
            while True:
                cs = cur[cj][cop]
                # Same-job predecessor.
                if cop > 0 and cur[cj][cop - 1] + times[cj][cop - 1] == cs:
                    cop = cop - 1
                    critical_path.append((cj, cop))
                    continue
                # Same-machine predecessor.
                mid = machines[cj][cop]
                pred = None
                for (i2, o2) in machine_to_ops[mid]:
                    if (i2, o2) == (cj, cop):
                        continue
                    if cur[i2][o2] + times[i2][o2] == cs:
                        pred = (i2, o2)
                        break
                if pred is not None:
                    cj, cop = pred
                    critical_path.append((cj, cop))
                    if cs == 0:
                        break
                    continue
                break  # reached start of path
            # Build current machine sequences once.
            seqs = _extract_machine_sequences(cur)
            # Try swapping each pair of adjacent critical ops on same machine.
            crit_set = set(critical_path)
            tried_any = False
            for mid in machine_ids:
                seq = seqs[mid]
                for k in range(len(seq) - 1):
                    j_a, j_b = seq[k], seq[k + 1]
                    # Op indices on this machine for these jobs.
                    op_a = None
                    op_b = None
                    for (i2, o2) in machine_to_ops[mid]:
                        if i2 == j_a and op_a is None:
                            op_a = o2
                        elif i2 == j_b and op_b is None:
                            op_b = o2
                    if op_a is None or op_b is None:
                        continue
                    if (j_a, op_a) not in crit_set or (j_b, op_b) not in crit_set:
                        continue
                    # Tentative swap.
                    new_seqs = {m: list(s) for m, s in seqs.items()}
                    new_seqs[mid][k], new_seqs[mid][k + 1] = (
                        new_seqs[mid][k + 1], new_seqs[mid][k]
                    )
                    tried_any = True
                    try:
                        new_st = simulate_active_schedule(new_seqs)
                    except ValueError:
                        continue
                    new_ms = _makespan(new_st)
                    if new_ms < cur_ms:
                        cur = new_st
                        cur_ms = new_ms
                        improved = True
                        break
                    if (time.time() - t0) >= time_limit_s - safety:
                        break
                if improved or (time.time() - t0) >= time_limit_s - safety:
                    break
            if not tried_any:
                break
        return cur

    # ==================================================================
    # (5) Heavy: exact ILP
    # ==================================================================
    def ilp_jssp(time_limit_s: float = 30.0) -> Optional[list[list[int]]]:
        """Solve the JSSP disjunctive ILP exactly with python-mip / CBC.

        Returns optimal-or-best-found start_times (2D list) within time_limit_s,
        or None if no feasible solution is reached. For small instances
        (n_jobs <= ~10, n_machines <= ~10) this typically finds the optimum.
        For Taillard-size benchmarks (15+ jobs) it usually runs out of time
        and returns a feasible-but-suboptimal solution.
        """
        try:
            from mip import (Model, INTEGER, BINARY, MINIMIZE, xsum,
                             OptimizationStatus)
        except ImportError:
            return None
        # Big-M based on total work upper bound (sum of all processing times).
        total_p = sum(int(times[i][j]) for i in range(nj) for j in range(nm))
        BIGM = total_p + 1
        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)
        # Variables: start_times s[i][j], makespan Cmax, disjunctive y for each
        # pair of ops on same machine.
        s = [[m.add_var(var_type=INTEGER, lb=0, ub=BIGM, name=f"s_{i}_{j}")
              for j in range(nm)] for i in range(nj)]
        Cmax = m.add_var(var_type=INTEGER, lb=0, ub=BIGM, name="Cmax")
        m.objective = Cmax
        # Precedence within each job.
        for i in range(nj):
            for j in range(1, nm):
                m += s[i][j] >= s[i][j - 1] + int(times[i][j - 1])
        # Cmax >= completion of every job's last op.
        for i in range(nj):
            m += Cmax >= s[i][nm - 1] + int(times[i][nm - 1])
        # Disjunctive constraints per machine.
        for mid in machine_ids:
            ops = machine_to_ops[mid]
            for a in range(len(ops)):
                ia, ja = ops[a]
                for b in range(a + 1, len(ops)):
                    ib, jb = ops[b]
                    y = m.add_var(var_type=BINARY, name=f"y_{ia}_{ja}_{ib}_{jb}")
                    # Either (ia,ja) before (ib,jb) or vice versa.
                    m += s[ia][ja] + int(times[ia][ja]) <= s[ib][jb] + BIGM * (1 - y)
                    m += s[ib][jb] + int(times[ib][jb]) <= s[ia][ja] + BIGM * y
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
        "machine_of": machine_of,
        "total_work": total_work,
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "validate_partial": validate_partial,
        "job_completion_time": job_completion_time,
        "machine_load": machine_load,
        "spt_dispatch": spt_dispatch,
        "simulate_active_schedule": simulate_active_schedule,
        "apply_critical_path_swap": apply_critical_path_swap,
        "ilp_jssp": ilp_jssp,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries (O(1) / O(nm)) -----
    {
        "name": "processing_time",
        "input": "j: int, op: int",
        "output": "int",
        "purpose": (
            "Processing time of job j's op-th operation. O(1) lookup. "
            "Cheaper than indexing instance['times'] manually because it "
            "validates bounds."
        ),
    },
    {
        "name": "machine_of",
        "input": "j: int, op: int",
        "output": "int",
        "purpose": (
            "Machine id (1-indexed) that job j's op-th operation runs on. "
            "O(1) lookup. Use when computing per-machine constraints."
        ),
    },
    {
        "name": "total_work",
        "input": "j: int",
        "output": "int",
        "purpose": (
            "Sum of processing times for all ops of job j (a lower bound on "
            "its completion time). O(n_machines)."
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
        "purpose": "Number of machines (also = ops per job in standard JSSP).",
    },
    # ----- Feasibility primitives -----
    {
        "name": "validate_partial",
        "input": "start_times: list[list[int]]",
        "output": "list[str]",
        "purpose": (
            "Returns every violated constraint (shape, non-negativity, job "
            "precedence, machine overlap) as a list of messages. Empty list "
            "means feasible. More informative than is_feasible's single-error "
            "result -- use inside local search to see all violations at once. "
            "O(n_jobs * n_machines + per-machine sort)."
        ),
    },
    {
        "name": "job_completion_time",
        "input": "j: int, start_times: list[list[int]]",
        "output": "int",
        "purpose": (
            "Completion time of job j: start_times[j][-1] + processing_time of "
            "its last op. No feasibility check. O(1)."
        ),
    },
    {
        "name": "machine_load",
        "input": "m: int, start_times: list[list[int]]",
        "output": "int",
        "purpose": (
            "Latest end-of-processing time on machine m under start_times "
            "(i.e., earliest moment m becomes free). Useful for greedy: a new "
            "op on m can't start before this. O(ops on machine m)."
        ),
    },
    # ----- Construction -----
    {
        "name": "spt_dispatch",
        "input": "(no args)",
        "output": "list[list[int]] (start_times)",
        "purpose": (
            "Shortest-Processing-Time dispatch rule. Simulate the schedule "
            "operation-by-operation: at each step, among jobs whose next op "
            "is ready, pick the smallest processing time. Always feasible. "
            "Good warm start. O((n_jobs * n_machines)^2)."
        ),
    },
    {
        "name": "simulate_active_schedule",
        "input": "machine_sequences: dict[int, list[int]]",
        "output": "list[list[int]] (start_times)",
        "purpose": (
            "Given each machine's processing order as a list of job ids, "
            "compute the implied start_times by event-driven simulation. "
            "Bridges sequence-style representations to start_times. Raises "
            "ValueError on malformed input or deadlock."
        ),
    },
    # ----- Improvement -----
    {
        "name": "apply_critical_path_swap",
        "input": "start_times: list[list[int]], time_limit_s: float = 5.0",
        "output": "list[list[int]] (start_times)",
        "purpose": (
            "Local search: find one critical-path block, swap adjacent ops "
            "on the same machine, re-simulate, keep iff makespan strictly "
            "decreases. First-improvement. Returns a NEW start_times. Caller "
            "must pass a feasible warm start (e.g., from spt_dispatch)."
        ),
    },
    # ----- Heavy: exact ILP -----
    {
        "name": "ilp_jssp",
        "input": "time_limit_s: float = 30.0",
        "output": "list[list[int]] | None (start_times)",
        "purpose": (
            "Solve the JSSP disjunctive ILP exactly with python-mip / CBC. "
            "Returns optimal-or-best-found start_times within time_limit_s, "
            "or None if no feasible solution. Effective on small instances "
            "(<= ~10x10); larger benchmarks usually time out with a feasible "
            "suboptimal solution -- still useful as a warm start."
        ),
    },
]
