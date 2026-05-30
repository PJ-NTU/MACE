"""Per-problem extras for CO-Bench Hybrid Reentrant Shop Scheduling.

Problem recap (see eval_func in data/.../config.py):
  Stage 1 (init):   each of n_jobs is initialized on one of n_machines identical
                    primary machines via LIST SCHEDULING in NATURAL job order.
                    init_time is the same for every job. -> the per-job machine
                    assignment is determined entirely by list scheduling.
  Stage 2 (setup):  jobs are processed on a single remote server in the order
                    given by the solver's `permutation`. A job's setup may start
                    only after its init completes AND the server is free.
  Stage 3 (main):   each job runs on the SAME machine that initialized it; on
                    each machine the order is the natural job-index order. A
                    job's main may start only after its setup completes AND its
                    machine is free.

Decision variable: `permutation` (1-indexed list of all jobs) -- the order in
which the remote server takes jobs. (The solution dict also carries
`batch_assignment`, but eval_func IGNORES it: machine assignment comes from the
deterministic init list-scheduling. We still emit a valid batch_assignment for
clarity / forward compatibility.)

Objective: minimize makespan = time the last main op finishes.

Tool groups:
  (1) Queries:           processing_time, setup_time, init_time_of,
                         n_jobs, n_machines, machine_of_job
  (2) Feasibility:       simulate_schedule, job_completion, stage_load
  (3) Construction/LS:   list_scheduling_priority, decode_priorities_to_schedule,
                         apply_local_swap
  (4) Heavy:             ilp_solve_small_instance
"""
from __future__ import annotations
import heapq
import time
from typing import Optional, Sequence


def extra_tools(instance: dict) -> dict:
    """Factory: returns HRSS-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench load_data):
      - n_jobs:           int
      - n_machines:       int   (number of primary machines)
      - init_time:        int   (per-job initialization time, constant)
      - setup_times:      list[int]   length n_jobs (server times, 0-indexed)
      - processing_times: list[int]   length n_jobs (main times, 0-indexed)
    """
    nj = int(instance["n_jobs"])
    nm = int(instance["n_machines"])
    it = int(instance["init_time"])
    st_list = [int(v) for v in instance["setup_times"]]
    pt_list = [int(v) for v in instance["processing_times"]]
    if len(st_list) != nj:
        raise ValueError(f"setup_times has {len(st_list)} entries, expected {nj}")
    if len(pt_list) != nj:
        raise ValueError(f"processing_times has {len(pt_list)} entries, expected {nj}")

    # Deterministic init list-scheduling: jobs 1..nj in natural order assigned
    # to least-loaded machine (tie: lowest id). Since init_time is identical for
    # every job, this resolves to round-robin: job index i (1-indexed) -> machine
    # ((i-1) % nm) + 1, and op1_finish[i] = it * ceil(i / nm).
    op1_finish: list[int] = [0] * (nj + 1)  # 1-indexed
    machine_of: list[int] = [0] * (nj + 1)
    _heap = [(0, mid) for mid in range(1, nm + 1)]
    heapq.heapify(_heap)
    for job in range(1, nj + 1):
        avail, mid = heapq.heappop(_heap)
        fin = avail + it
        op1_finish[job] = fin
        machine_of[job] = mid
        heapq.heappush(_heap, (fin, mid))

    # Precompute the natural-order job list per machine (for stage 3).
    jobs_by_machine: dict[int, list[int]] = {mid: [] for mid in range(1, nm + 1)}
    for j in range(1, nj + 1):
        jobs_by_machine[machine_of[j]].append(j)

    # ==================================================================
    # (1) Queries -- O(1)
    # ==================================================================
    def processing_time(j: int) -> int:
        """Main-stage processing time of job j (1-indexed)."""
        if not (1 <= j <= nj):
            raise ValueError(f"j={j} out of [1, {nj}]")
        return int(pt_list[j - 1])

    def setup_time(j: int) -> int:
        """Server (stage-2) setup time of job j (1-indexed)."""
        if not (1 <= j <= nj):
            raise ValueError(f"j={j} out of [1, {nj}]")
        return int(st_list[j - 1])

    def init_time_of(j: int) -> int:
        """Stage-1 initialization time of job j (constant across jobs)."""
        if not (1 <= j <= nj):
            raise ValueError(f"j={j} out of [1, {nj}]")
        return int(it)

    def n_jobs() -> int:
        return int(nj)

    def n_machines() -> int:
        return int(nm)

    def machine_of_job(j: int) -> int:
        """The primary machine job j is bound to (1-indexed). Deterministic;
        comes from list scheduling on identical init machines."""
        if not (1 <= j <= nj):
            raise ValueError(f"j={j} out of [1, {nj}]")
        return int(machine_of[j])

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _validate_permutation(permutation: Sequence[int]) -> list[int]:
        if len(permutation) != nj:
            raise ValueError(f"permutation length {len(permutation)} != n_jobs {nj}")
        try:
            perm_int = [int(p) for p in permutation]
        except Exception:
            raise ValueError("permutation must contain integers")
        if sorted(perm_int) != list(range(1, nj + 1)):
            raise ValueError(
                "permutation must be a 1-indexed permutation of 1..n_jobs"
            )
        return perm_int

    def simulate_schedule(permutation: Sequence[int]) -> dict:
        """Replay the full three-stage schedule for a given `permutation`.

        Returns a dict with:
          - 'op1_finish':  list[int] length nj+1 (1-indexed; index 0 is 0).
          - 'op2_start':   list[int] length nj+1 (server start per job)
          - 'op2_finish':  list[int] length nj+1 (server finish per job)
          - 'op3_start':   list[int] length nj+1 (main start per job)
          - 'op3_finish':  list[int] length nj+1 (main finish per job)
          - 'makespan':    int (= max(op3_finish))
          - 'machine_of':  list[int] length nj+1 (1-indexed mapping)

        Raises ValueError if `permutation` is not a valid 1..nj permutation.
        Faithful to config.eval_func; mirrors its semantics exactly. O(n_jobs).
        """
        perm = _validate_permutation(permutation)

        # Stage 2 on server.
        op2_start = [0] * (nj + 1)
        op2_finish = [0] * (nj + 1)
        server_t = 0
        for job in perm:
            start = max(op1_finish[job], server_t)
            fin = start + st_list[job - 1]
            op2_start[job] = start
            op2_finish[job] = fin
            server_t = fin

        # Stage 3 per machine in natural job-order.
        op3_start = [0] * (nj + 1)
        op3_finish = [0] * (nj + 1)
        for mid in range(1, nm + 1):
            cur = 0
            for job in jobs_by_machine[mid]:
                start = max(cur, op2_finish[job])
                fin = start + pt_list[job - 1]
                op3_start[job] = start
                op3_finish[job] = fin
                cur = fin

        makespan = max(op3_finish[1:]) if nj > 0 else 0
        return {
            "op1_finish": list(op1_finish),
            "op2_start": op2_start,
            "op2_finish": op2_finish,
            "op3_start": op3_start,
            "op3_finish": op3_finish,
            "makespan": int(makespan),
            "machine_of": list(machine_of),
        }

    def job_completion(j: int, permutation: Sequence[int]) -> int:
        """Stage-3 finish time of job j under the given permutation."""
        if not (1 <= j <= nj):
            raise ValueError(f"j={j} out of [1, {nj}]")
        sim = simulate_schedule(permutation)
        return int(sim["op3_finish"][j])

    def stage_load(stage: int, permutation: Sequence[int],
                   machine: Optional[int] = None) -> int:
        """Latest end-of-busy time of a stage's resource.
          stage=1: max(op1_finish), or end-of-busy on `machine` (1..nm)
          stage=2: server finish time (= max op2_finish)
          stage=3: max(op3_finish), or end-of-busy on `machine` (1..nm)
        For stage 1, since init is deterministic, `permutation` is ignored.
        """
        if stage not in (1, 2, 3):
            raise ValueError(f"stage must be 1, 2, or 3, got {stage}")
        if stage == 1:
            if machine is None:
                return int(max(op1_finish[1:])) if nj > 0 else 0
            if not (1 <= machine <= nm):
                raise ValueError(f"machine={machine} out of [1, {nm}]")
            jobs = jobs_by_machine[machine]
            return int(max((op1_finish[j] for j in jobs), default=0))
        sim = simulate_schedule(permutation)
        if stage == 2:
            return int(max(sim["op2_finish"][1:])) if nj > 0 else 0
        # stage == 3
        if machine is None:
            return int(sim["makespan"])
        if not (1 <= machine <= nm):
            raise ValueError(f"machine={machine} out of [1, {nm}]")
        jobs = jobs_by_machine[machine]
        return int(max((sim["op3_finish"][j] for j in jobs), default=0))

    # ==================================================================
    # (3) Construction / improvement
    # ==================================================================
    def list_scheduling_priority(rule: str = "spt_setup") -> list[int]:
        """Build a server permutation from a priority rule. Rules:
          - 'natural':    1, 2, ..., n_jobs
          - 'spt_setup':  shortest setup time first
          - 'lpt_setup':  longest setup time first
          - 'spt_main':   shortest main processing time first
          - 'lpt_main':   longest main processing time first
          - 'spt_total':  shortest setup+main first
          - 'lpt_total':  longest setup+main first
          - 'erd':        earliest release date (op1_finish) first
                          (= natural order for round-robin init, but
                          ties broken by setup time ascending)
        Ties always broken by job index ascending. Returns a 1-indexed
        permutation of length n_jobs.
        """
        idx = list(range(1, nj + 1))
        if rule == "natural":
            return idx
        if rule == "spt_setup":
            return sorted(idx, key=lambda j: (st_list[j - 1], j))
        if rule == "lpt_setup":
            return sorted(idx, key=lambda j: (-st_list[j - 1], j))
        if rule == "spt_main":
            return sorted(idx, key=lambda j: (pt_list[j - 1], j))
        if rule == "lpt_main":
            return sorted(idx, key=lambda j: (-pt_list[j - 1], j))
        if rule == "spt_total":
            return sorted(idx, key=lambda j: (st_list[j - 1] + pt_list[j - 1], j))
        if rule == "lpt_total":
            return sorted(idx, key=lambda j: (-(st_list[j - 1] + pt_list[j - 1]), j))
        if rule == "erd":
            return sorted(idx, key=lambda j: (op1_finish[j], st_list[j - 1], j))
        raise ValueError(f"unknown priority rule: {rule!r}")

    def decode_priorities_to_schedule(priorities: Sequence[float]) -> dict:
        """Convert a length-n_jobs priority vector (lower = higher priority,
        i.e., scheduled earlier on the server) into a full schedule via
        simulate_schedule. Ties are broken by job index ascending. Returns the
        same dict shape as simulate_schedule plus the resolved 'permutation'.

        Useful when the caller searches in continuous space (e.g., a CMA-ES
        priority vector) and needs to evaluate solutions.
        """
        if len(priorities) != nj:
            raise ValueError(
                f"priorities length {len(priorities)} != n_jobs {nj}"
            )
        order = sorted(range(1, nj + 1), key=lambda j: (float(priorities[j - 1]), j))
        sim = simulate_schedule(order)
        sim["permutation"] = order
        return sim

    def apply_local_swap(permutation: Sequence[int],
                         time_limit_s: float = 5.0,
                         neighbourhood: str = "adjacent") -> list[int]:
        """First-improvement local search on the server `permutation`.

        Neighbourhoods:
          - 'adjacent': swap positions (k, k+1)            (O(n) moves)
          - 'swap':     swap any two positions (k, l)      (O(n^2) moves)
          - 'insert':   move position k to position l      (O(n^2) moves)

        Re-simulates the full schedule for each tentative move; the move is
        accepted iff it strictly decreases the makespan. Self-monitors
        `time_limit_s`. Returns a NEW permutation (does not mutate the input).
        """
        if neighbourhood not in ("adjacent", "swap", "insert"):
            raise ValueError(
                f"neighbourhood must be 'adjacent'|'swap'|'insert', got {neighbourhood!r}"
            )
        cur = list(_validate_permutation(permutation))
        cur_ms = int(simulate_schedule(cur)["makespan"])
        t0 = time.time()
        safety = 0.05
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            if neighbourhood == "adjacent":
                moves = [(k, k + 1) for k in range(nj - 1)]
            elif neighbourhood == "swap":
                moves = [(k, l) for k in range(nj) for l in range(k + 1, nj)]
            else:  # insert
                moves = [(k, l) for k in range(nj) for l in range(nj) if k != l]
            for (k, l) in moves:
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                cand = list(cur)
                if neighbourhood in ("adjacent", "swap"):
                    cand[k], cand[l] = cand[l], cand[k]
                else:
                    job = cand.pop(k)
                    cand.insert(l, job)
                ms = int(simulate_schedule(cand)["makespan"])
                if ms < cur_ms:
                    cur = cand
                    cur_ms = ms
                    improved = True
                    break
        return cur

    # ==================================================================
    # (4) Heavy: exact ILP for small instances
    # ==================================================================
    def ilp_solve_small_instance(time_limit_s: float = 30.0
                                 ) -> Optional[list[int]]:
        """Solve the HRSS to optimality (within time_limit_s) using a MIP
        formulation in python-mip / CBC. Returns the optimal `permutation`
        (1-indexed list of length n_jobs), or None if no feasible solution was
        found.

        Variables:
          - y[i][k] in {0,1}: 1 iff job i is the k-th job processed on server.
          - s2[i] >= 0:       server start time of job i.
          - s3[i] >= 0:       main start time of job i.
          - Cmax >= 0.
        Constraints:
          - Each job assigned to exactly one server position; each position
            holds exactly one job (assignment polytope).
          - Server start order encoded via positional precedence: if job i is
            at position k and job j is at position k+1, then s2[j] >= s2[i] +
            setup[i] (linearised with big-M and two y vars).
          - s2[i] >= op1_finish[i] (release-date from deterministic init).
          - s3[i] >= s2[i] + setup[i].
          - Per-machine natural-order chain: for jobs i < j on same machine,
            s3[j] >= s3[i] + proc[i].
          - Cmax >= s3[i] + proc[i] for every i.

        Practical: works well up to ~15-20 jobs. Beyond that CBC will likely
        time out but may still return a feasible suboptimal permutation.
        """
        try:
            from mip import (
                Model, BINARY, CONTINUOUS, MINIMIZE, xsum, OptimizationStatus
            )
        except ImportError:
            return None

        # Big-M based on a loose upper bound on the makespan.
        BIGM = (
            it * nj                       # all inits in series (very loose)
            + sum(st_list)                # all setups in series
            + sum(pt_list)                # all main ops in series
            + 1
        )

        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)

        # y[i][k] 1-indexed: i in 1..nj job, k in 1..nj position.
        y = [[m.add_var(var_type=BINARY, name=f"y_{i}_{k}")
              for k in range(nj + 1)]
             for i in range(nj + 1)]
        s2 = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub=BIGM, name=f"s2_{i}")
              for i in range(nj + 1)]
        s3 = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub=BIGM, name=f"s3_{i}")
              for i in range(nj + 1)]
        Cmax = m.add_var(var_type=CONTINUOUS, lb=0.0, ub=BIGM, name="Cmax")
        m.objective = Cmax

        # Assignment constraints.
        for i in range(1, nj + 1):
            m += xsum(y[i][k] for k in range(1, nj + 1)) == 1
        for k in range(1, nj + 1):
            m += xsum(y[i][k] for i in range(1, nj + 1)) == 1

        # Release dates from deterministic init.
        for i in range(1, nj + 1):
            m += s2[i] >= int(op1_finish[i])

        # Server precedence between consecutive positions:
        # for every (i, j, k) with i != j: if y[i][k]=1 and y[j][k+1]=1 then
        # s2[j] >= s2[i] + setup[i]. Linearised with big-M:
        # s2[j] >= s2[i] + setup[i] - BIGM*(2 - y[i][k] - y[j][k+1]).
        for k in range(1, nj):
            for i in range(1, nj + 1):
                for j in range(1, nj + 1):
                    if i == j:
                        continue
                    m += s2[j] >= s2[i] + int(st_list[i - 1]) \
                        - BIGM * (2 - y[i][k] - y[j][k + 1])

        # Setup -> main precedence per job.
        for i in range(1, nj + 1):
            m += s3[i] >= s2[i] + int(st_list[i - 1])

        # Per-machine natural-order chain on stage 3.
        for mid in range(1, nm + 1):
            seq = jobs_by_machine[mid]
            for a in range(len(seq) - 1):
                i = seq[a]
                j = seq[a + 1]
                m += s3[j] >= s3[i] + int(pt_list[i - 1])

        # Makespan.
        for i in range(1, nj + 1):
            m += Cmax >= s3[i] + int(pt_list[i - 1])

        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL,
                          OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None

        # Decode permutation from y.
        permutation = [0] * nj
        for k in range(1, nj + 1):
            chosen = None
            best_val = -1.0
            for i in range(1, nj + 1):
                v = y[i][k].x
                if v is not None and v > best_val:
                    best_val = float(v)
                    chosen = i
            if chosen is None:
                return None
            permutation[k - 1] = int(chosen)
        if sorted(permutation) != list(range(1, nj + 1)):
            # Numerical fallback: solver returned something inconsistent.
            return None
        return permutation

    return {
        "processing_time": processing_time,
        "setup_time": setup_time,
        "init_time_of": init_time_of,
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "machine_of_job": machine_of_job,
        "simulate_schedule": simulate_schedule,
        "job_completion": job_completion,
        "stage_load": stage_load,
        "list_scheduling_priority": list_scheduling_priority,
        "decode_priorities_to_schedule": decode_priorities_to_schedule,
        "apply_local_swap": apply_local_swap,
        "ilp_solve_small_instance": ilp_solve_small_instance,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "processing_time",
        "input": "j: int   (1-indexed job id)",
        "output": "int",
        "purpose": (
            "Main-stage processing time of job j. O(1). 1-indexed throughout "
            "this problem to match CO-Bench's permutation convention."
        ),
    },
    {
        "name": "setup_time",
        "input": "j: int   (1-indexed)",
        "output": "int",
        "purpose": "Server (stage-2) setup time of job j. O(1).",
    },
    {
        "name": "init_time_of",
        "input": "j: int   (1-indexed)",
        "output": "int",
        "purpose": (
            "Stage-1 initialization time of job j. Constant across jobs by "
            "construction, but exposed per-job for symmetry. O(1)."
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
        "purpose": "Number of primary machines (used for both stage 1 and stage 3).",
    },
    {
        "name": "machine_of_job",
        "input": "j: int   (1-indexed)",
        "output": "int   (1-indexed machine id)",
        "purpose": (
            "Primary machine that job j is bound to. DETERMINISTIC: comes from "
            "list scheduling on identical init machines, independent of the "
            "solver's `batch_assignment` (which eval_func ignores). With "
            "constant init_time this is just round-robin: job j -> machine "
            "((j-1) % n_machines) + 1."
        ),
    },
    # ----- Feasibility primitives -----
    {
        "name": "simulate_schedule",
        "input": "permutation: list[int]   (1-indexed permutation of 1..n_jobs)",
        "output": "dict with keys op1_finish, op2_start, op2_finish, op3_start, "
                  "op3_finish, makespan, machine_of",
        "purpose": (
            "Replay the full three-stage HRSS schedule for the given server "
            "`permutation`. Mirrors config.eval_func exactly. Use this to "
            "compute makespan and per-job timings without rebuilding the "
            "feasibility scaffolding by hand. Raises ValueError on invalid "
            "permutations. O(n_jobs)."
        ),
    },
    {
        "name": "job_completion",
        "input": "j: int   (1-indexed),   permutation: list[int]",
        "output": "int",
        "purpose": (
            "Stage-3 finish time of job j under the given permutation. Thin "
            "wrapper around simulate_schedule."
        ),
    },
    {
        "name": "stage_load",
        "input": "stage: int in {1, 2, 3},   permutation: list[int],   machine: int | None = None",
        "output": "int",
        "purpose": (
            "Latest end-of-busy time of a stage's resource. stage=1 ignores "
            "permutation (init is deterministic); stage=2 returns the server "
            "finish time; stage=3 returns either the global makespan (machine "
            "= None) or the end-of-busy of one primary machine. Useful for "
            "diagnosing bottlenecks (server vs. a specific machine)."
        ),
    },
    # ----- Construction / local search -----
    {
        "name": "list_scheduling_priority",
        "input": "rule: str = 'spt_setup'",
        "output": "list[int]   (1-indexed permutation)",
        "purpose": (
            "Build a server permutation from a priority rule. Supported rules: "
            "'natural', 'spt_setup', 'lpt_setup', 'spt_main', 'lpt_main', "
            "'spt_total', 'lpt_total', 'erd' (earliest release-date). Each "
            "rule yields one candidate -- try several and keep the best as a "
            "warm start. O(n_jobs log n_jobs)."
        ),
    },
    {
        "name": "decode_priorities_to_schedule",
        "input": "priorities: list[float]   (length n_jobs; lower = earlier on server)",
        "output": "dict   (simulate_schedule output + key 'permutation')",
        "purpose": (
            "Convert a length-n_jobs priority vector into a server permutation "
            "(stable tie-breaking by job index) and simulate it. Useful when "
            "the caller searches in a continuous space (e.g., CMA-ES, random "
            "perturbations of a base priority vector)."
        ),
    },
    {
        "name": "apply_local_swap",
        "input": "permutation: list[int], time_limit_s: float = 5.0, "
                 "neighbourhood: str = 'adjacent'   ('adjacent'|'swap'|'insert')",
        "output": "list[int]   (improved permutation)",
        "purpose": (
            "First-improvement local search on the server permutation. Each "
            "tentative neighbour is evaluated with simulate_schedule and kept "
            "iff makespan strictly decreases. Self-monitors time_limit_s. "
            "'adjacent' is the fastest (O(n) moves per pass); 'insert' is the "
            "most thorough. Returns a NEW permutation."
        ),
    },
    # ----- Heavy: exact ILP -----
    {
        "name": "ilp_solve_small_instance",
        "input": "time_limit_s: float = 30.0",
        "output": "list[int] | None   (optimal permutation, or None)",
        "purpose": (
            "Solve the HRSS to optimality (within time_limit_s) using a MIP "
            "formulation in python-mip / CBC. Variables: assignment y[i][k], "
            "server starts s2, main starts s3, Cmax. Works well up to "
            "~15-20 jobs; beyond that CBC usually times out but may still "
            "return a feasible suboptimal permutation. Returns None on "
            "solver failure or no feasible solution."
        ),
    },
]
