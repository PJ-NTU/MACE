"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    if "warehouse_open" not in solution:
        return False, "solution missing 'warehouse_open' key"
    if "assignments" not in solution:
        return False, "solution missing 'assignments' key"

    warehouse_open = solution["warehouse_open"]
    assignments = solution["assignments"]

    if not isinstance(warehouse_open, (list, tuple)):
        return False, f"'warehouse_open' must be list, got {type(warehouse_open).__name__}"
    if len(warehouse_open) != m:
        return False, f"'warehouse_open' length {len(warehouse_open)} != m={m}"

    if not isinstance(assignments, (list, tuple)):
        return False, f"'assignments' must be list, got {type(assignments).__name__}"
    if len(assignments) != n:
        return False, f"'assignments' length {len(assignments)} != n={n}"

    for j in range(n):
        row = assignments[j]
        if not isinstance(row, (list, tuple)):
            return False, f"assignments[{j}] must be list, got {type(row).__name__}"
        if len(row) != m:
            return False, f"assignments[{j}] length {len(row)} != m={m}"

        for i in range(m):
            allocation = row[i]
            if not (abs(allocation) < 1e-6 or abs(allocation - 1.0) < 1e-6):
                return False, f"customer {j} has non-binary assignment value {allocation} for warehouse {i}"

        assigned_sum = sum(row)
        if abs(assigned_sum - 1.0) > 1e-6:
            return False, f"customer {j} assignment sum {assigned_sum} != 1"

        for i in range(m):
            if row[i] > 0 and warehouse_open[i] != 1:
                return False, f"customer {j} assigned to closed warehouse {i}"

    return True, None
'''
