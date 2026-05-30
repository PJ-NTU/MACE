"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    start_times = solution.get("start_times")
    if start_times is None:
        return False, "solution missing 'start_times' key"
    if not isinstance(start_times, (list, tuple)):
        return False, f"'start_times' must be list, got {type(start_times).__name__}"

    if len(start_times) != n_jobs:
        return False, f"start_times has {len(start_times)} rows, expected {n_jobs}"

    for i, row in enumerate(start_times):
        if not isinstance(row, (list, tuple)):
            return False, f"start_times[{i}] must be list, got {type(row).__name__}"
        if len(row) != n_machines:
            return False, f"start_times[{i}] has {len(row)} entries, expected {n_machines}"
        for j, t in enumerate(row):
            if not isinstance(t, int):
                return False, f"start_times[{i}][{j}] must be int, got {type(t).__name__}"
            if t < 0:
                return False, f"start_times[{i}][{j}] is negative ({t})"

    # Check non-parallel operation: operations within the same job must not overlap
    for i in range(n_jobs):
        ops = sorted(
            [(start_times[i][j], start_times[i][j] + times[i][j]) for j in range(n_machines)],
            key=lambda x: x[0]
        )
        for k in range(len(ops) - 1):
            if ops[k][1] > ops[k + 1][0]:
                return False, f"Overlapping operations for job {i}: {ops[k]} and {ops[k+1]}"

    # Check machine exclusivity: operations on the same machine must not overlap
    machine_schedules = {}
    for i in range(n_jobs):
        for j in range(n_machines):
            machine_id = machines[i][j]
            st = start_times[i][j]
            ft = st + times[i][j]
            if machine_id not in machine_schedules:
                machine_schedules[machine_id] = []
            machine_schedules[machine_id].append((st, ft, i, j))

    for machine_id, ops in machine_schedules.items():
        ops_sorted = sorted(ops, key=lambda x: x[0])
        for k in range(1, len(ops_sorted)):
            prev_st, prev_ft, prev_job, prev_op = ops_sorted[k - 1]
            curr_st, curr_ft, curr_job, curr_op = ops_sorted[k]
            if prev_ft > curr_st:
                return False, (
                    f"Machine {machine_id}: job {prev_job} op {prev_op} "
                    f"(finishes {prev_ft}) overlaps job {curr_job} op {curr_op} "
                    f"(starts {curr_st})"
                )

    return True, None
'''
